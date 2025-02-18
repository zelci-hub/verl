# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import asyncio
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import uuid
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
import numpy as np
from copy import deepcopy

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import AsyncLLMEngine, LLM, SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from verl.third_party.vllm import vllm_version

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

# Monkey patch for AsyncLLMEngine
import time
def sleep(self, level: int = 1):
        """
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        :param level: The sleep level. Level 1 sleep will offload the model 
            weights and discard the kv cache. The content of kv cache is 
            forgotten. Level 1 sleep is good for sleeping and waking up the 
            engine to run the same model again. The model weights are backed 
            up in CPU memory. Please make sure there's enough CPU memory to 
            store the model weights. Level 2 sleep will discard both the model 
            weights and the kv cache. The content of both the model weights 
            and kv cache is forgotten. Level 2 sleep is good for sleeping and 
            waking up the engine to run a different model or update the model, 
            where previous model weights are not needed. It reduces CPU memory 
            pressure.
        """
        self.engine.reset_prefix_cache()
        self.engine.sleep(level=level)

def wake_up(self):
    """
    Wake up the engine from sleep mode. See the :meth:`sleep` method
    for more details."""
    self.engine.wake_up()

AsyncLLMEngine.sleep = sleep
AsyncLLMEngine.wake_up = wake_up



class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        if config.async_engine:
            self.inference_engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(
                    model=model_path,
                    enable_sleep_mode=True,
                    tensor_parallel_size=tensor_parallel_size,
                    distributed_executor_backend="external_launcher",
                    dtype=config.dtype,
                    enforce_eager=config.enforce_eager,
                    gpu_memory_utilization=config.gpu_memory_utilization,
                    disable_custom_all_reduce=True,
                    skip_tokenizer_init=False,
                    max_model_len=config.prompt_length + config.response_length,
                    disable_log_stats=config.disable_log_stats,
                    max_num_batched_tokens=max_num_batched_tokens,
                    enable_chunked_prefill=config.enable_chunked_prefill,
                )
            )
        else:     
            self.inference_engine = LLM(
                model=model_path,
                enable_sleep_mode=True,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="external_launcher",
                dtype=config.dtype,
                enforce_eager=config.enforce_eager,
                gpu_memory_utilization=config.gpu_memory_utilization,
                disable_custom_all_reduce=True,
                skip_tokenizer_init=False,
                max_model_len=config.prompt_length + config.response_length,
                disable_log_stats=config.disable_log_stats,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=config.enable_chunked_prefill,
            )
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        
        if prompts.meta_info.get('val_temperature', None):
            kwargs['temperature'] = prompts.meta_info['val_temperature']
        
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            if self.config.async_engine:
                outputs = []
                for output in self.generate_sequences_fn(prompts):
                    outputs.append(output)
                return DataProto.concat(outputs)
            else:
                outputs = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=idx_list,
                    use_tqdm=False)
        
        # Extract token IDs and log probabilities
        response = []
        log_probs = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)
                if hasattr(output.outputs[sample_id], 'logprobs') and output.outputs[sample_id].logprobs is not None:
                    # Directly use the list of floats returned by vLLM
                    log_prob_list = []
                    for log_prob in output.outputs[sample_id].logprobs:
                        log_prob_list.append(next(iter(log_prob.values())).logprob)
                    log_probs.append(log_prob_list)
                else:
                    log_probs.append([0.0] * len(output.outputs[sample_id].token_ids))
        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)
        log_probs = pad_2d_list_to_length(log_probs, 0.0,
                                         max_length=self.config.response_length).to(idx.device)

        non_tensor_batch = deepcopy(prompts.non_tensor_batch)
        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
            # Create interleaved non_tensor_batch
            non_tensor_batch = {}
            for key, val in prompts.non_tensor_batch.items():
                # Repeat each element n times (interleaved)
                repeated_val = np.repeat(val, self.config.n)
                non_tensor_batch[key] = repeated_val

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)
        if self.config.vllm_log_prob:
            batch['old_log_probs'] = log_probs
        
        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=prompts.meta_info
        )

    @torch.no_grad()
    def generate_sequences_fn(self, prompts: DataProto, **kwargs):
        """Generator function that yields outputs as they complete (may be out of order).
        
        Args:
            prompts: DataProto containing the input prompts
            **kwargs: Additional arguments to modify sampling parameters
        
        Yields:
            tuple: (prompt_idx, DataProto) containing the original prompt index and its generated sequence
        """
        assert self.config.async_engine, "generate_sequences_fn requires async_engine=True"
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        if prompts.meta_info.get('val_temperature', None):
            kwargs['temperature'] = prompts.meta_info['val_temperature']

        self.update_sampling_params(**kwargs)

        async def _create_task(prompt_idx, task):
            """Process a single generation task and return its result with original index"""
            async for output in task:
                last_output = output
            return prompt_idx, last_output

        async def _async_generate():
            # Create all tasks
            tasks = [
                _create_task(
                    prompt_idx,
                    self.inference_engine.generate(
                        prompt=TokensPrompt(prompt_token_ids=prompt_tokens),
                        sampling_params=self.sampling_params,
                        request_id=str(uuid.uuid4()),
                    )
                ) for prompt_idx, prompt_tokens in enumerate(idx_list)
            ]
            
            # Use as_completed to yield results as they finish
            for completed_task in asyncio.as_completed(tasks):
                prompt_idx, output = await completed_task
                
                # Process output
                response = []
                log_probs = []
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    if hasattr(output.outputs[sample_id], 'logprobs') and output.outputs[sample_id].logprobs is not None:
                        log_prob_list = []
                        for log_prob in output.outputs[sample_id].logprobs:
                            log_prob_list.append(next(iter(log_prob.values())).logprob)
                        log_probs.append(log_prob_list)
                    else:
                        log_probs.append([0.0] * len(output.outputs[sample_id].token_ids))

                response = pad_2d_list_to_length(response, self.pad_token_id,
                                               max_length=self.config.response_length).to(idx.device)
                log_probs = pad_2d_list_to_length(log_probs, 0.0,
                                               max_length=self.config.response_length).to(idx.device)

                # Process single sequence
                single_idx = idx[prompt_idx:prompt_idx+1]
                single_attention_mask = attention_mask[prompt_idx:prompt_idx+1]
                single_position_ids = position_ids[prompt_idx:prompt_idx+1]
                non_tensor_batch = {}
                for key, val in prompts.non_tensor_batch.items():
                    # Get single value and repeat n times
                    single_val = val[prompt_idx:prompt_idx+1]
                    non_tensor_batch[key] = single_val
                # Handle multiple samples per prompt when n > 1 and sampling
                if self.config.n > 1 and do_sample:
                    single_idx = single_idx.repeat_interleave(self.config.n, dim=0)
                    single_attention_mask = single_attention_mask.repeat_interleave(self.config.n, dim=0)
                    single_position_ids = single_position_ids.repeat_interleave(self.config.n, dim=0)
                    # Create interleaved non_tensor_batch for this subset
                    for key, val in non_tensor_batch.items():
                        # Get single value and repeat n times
                        repeated_val = np.repeat(val, self.config.n)
                        non_tensor_batch[key] = repeated_val

                seq = torch.cat([single_idx, response], dim=-1)
                
                response_length = response.size(1)
                delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).repeat(response.size(0), 1)
                
                response_position_ids = single_position_ids[:, -1:] + delta_position_id
                position_ids_out = torch.cat([single_position_ids, response_position_ids], dim=-1)
                response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
                attention_mask_out = torch.cat((single_attention_mask, response_attention_mask), dim=-1)

                batch = TensorDict(
                    {
                        'prompts': single_idx,
                        'responses': response,
                        'input_ids': seq,
                        'attention_mask': attention_mask_out,
                        'position_ids': position_ids_out
                    },
                    batch_size=response.size(0))
                if self.config.vllm_log_prob:
                    batch['old_log_probs'] = log_probs
                final_batch = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                    meta_info=prompts.meta_info
                )
                yield final_batch

        # Create new event loop for this generator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create async generator
            async_gen = _async_generate()
            
            while True:
                try:
                    # Run until next result is ready
                    result = loop.run_until_complete(async_gen.__anext__())
                    yield result
                except StopAsyncIteration:
                    break
                except Exception as e:
                    # Ensure loop is cleaned up on error
                    if not loop.is_closed():
                        loop.close()
                    raise e
        finally:
            # Clean up
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            if not loop.is_closed():
                # Run loop one final time to execute any remaining callbacks
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        return None
