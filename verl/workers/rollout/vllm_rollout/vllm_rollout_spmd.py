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
import copy
import os
import numpy as np
from typing import List
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import AsyncGenerator, Generator, List, Tuple, TypeVar, Union
import asyncio
import numpy as np
import torch
import torch.distributed
import uuid
from omegaconf import DictConfig
from tensordict import TensorDict

from rllm.environments.tools import PythonInterpreter, ToolCaller
from torch import nn
from typing import Any, Dict, Union
from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.utils import run_async_generator

from vllm import AsyncLLMEngine, LLM, SamplingParams, TokensPrompt
from vllm.distributed import parallel_state as vllm_ps
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput

# Set very long timeout to effectively disable it
os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '1000000000'  # 1e9 seconds

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

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


# Monkey patch for AsyncLLMEngine to support sleep and wakeup operations.
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

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, reward_fn, val_reward_fn, **kwargs):
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
        
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        
        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
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
                    enable_prefix_caching=True
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
                enable_prefix_caching=True,
            )
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
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
        self.tokenizer = tokenizer
        self.interpreter = PythonInterpreter(n_sandboxes=2)
        self.tool_caller = ToolCaller(tools=[self.interpreter], parser_type="python")

    async def generate_sequence_task(self, idx: int, prompt_tokens: Union[Dict[str, Any], List[int]], sampling_params: SamplingParams = None) -> Tuple[int, RequestOutput]:
        """Generate a sequence asynchronously using vLLM.
        
        This method creates an asynchronous task for generating text from the given prompt tokens.
        It streams the outputs and returns the final result along with the original prompt index.
        
        Args:
            idx: The index of the prompt in the original batch
            prompt_tokens: List of token IDs representing the input prompt
            sampling_params: Optional custom sampling parameters, defaults to self.sampling_params
            
        Returns:
            tuple: (idx, last_output) where idx is the original prompt index and 
                  last_output contains the generated sequence and related information
        """
        if sampling_params is None:
            sampling_params = self.sampling_params
        if isinstance(prompt_tokens, list):
            prompt_tokens = {'prompt_token_ids': prompt_tokens}
        task = self.inference_engine.generate(
                    prompt=TokensPrompt(**prompt_tokens),
                    sampling_params=sampling_params,
                    request_id=str(uuid.uuid4()),
        )
        async for output in task:
            last_output = output
        return idx, last_output

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
        # Handle async engine case
        if self.config.async_engine:
            outputs = []
            for output in self.generate_sequences_async(prompts):
                outputs.append(output)
            return DataProto.concat(outputs)
        # Rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        if prompts.meta_info.get('agent_rollout', False):
            kwargs['n'] = 1

        if is_validate:
            # TODO: try **
            kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            })

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)
        
        # Extract token IDs and log probabilities
        response = []
        log_probs = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)
                if self.config.vllm_log_prob and hasattr(output.outputs[sample_id], 'logprobs') and output.outputs[sample_id].logprobs is not None:
                    # Directly use the list of floats returned by vLLM
                    log_prob_list = []
                    for log_prob in output.outputs[sample_id].logprobs:
                        log_prob_list.append(next(iter(log_prob.values())).logprob)
                    log_probs.append(log_prob_list)
                else:
                    log_probs.append([0.0] * len(output.outputs[sample_id].token_ids))
        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)
        if self.config.vllm_log_prob:
            log_probs = pad_2d_list_to_length(log_probs, 0.0,
                                            max_length=self.config.response_length).to(idx.device)

        non_tensor_batch = deepcopy(prompts.non_tensor_batch)
        if self.sampling_params.n > 1 and do_sample and not prompts.meta_info.get('agent_rollout', False):
            idx = _repeat_interleave(idx, self.sampling_params.n)
            attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
            position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            batch_size = batch_size * self.sampling_params.n
            # Create interleaved non_tensor_batch
            non_tensor_batch = {}
            for key, val in prompts.non_tensor_batch.items():
                # Repeat each element n times (interleaved)
                non_tensor_batch[key] = _repeat_interleave(val, self.sampling_params.n)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

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
        
        if self.reward_fn is not None and not is_validate:
            init_batch = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                meta_info=prompts.meta_info
            )
            reward_tensor = self.reward_fn(init_batch)
            batch['token_level_scores'] = reward_tensor
        
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=prompts.meta_info
        )

    @torch.no_grad()
    def generate_sequences_async(self, prompts: DataProto, **kwargs):
        """Generator function that yields outputs as they complete (may be out of order).
        
        Args:
            prompts: DataProto containing the input prompts
            **kwargs: Additional arguments to modify sampling parameters
        
        Yields:
            tuple: (prompt_idx, DataProto) containing the original prompt index and its generated sequence
        """
        assert self.config.async_engine, "generate_sequences_async requires async_engine=True"
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
            
        if prompts.meta_info.get('agent_rollout', False):
            kwargs['n'] = 1
        
        if is_validate:
            # TODO: try **
            kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            })


        updated_sampling_params = deepcopy(self.sampling_params)
        for key, value in kwargs.items():
            if hasattr(updated_sampling_params, key):
                setattr(updated_sampling_params, key, value)

        async def _async_generate():
            # Create all tasks
            tasks = [
                self.generate_sequence_task(prompt_idx, vllm_inputs[prompt_idx], updated_sampling_params) for prompt_idx in range(batch_size)
            ]
            for completed_task in asyncio.as_completed(tasks):
                prompt_idx, output = await completed_task
                
                # Process output
                response = []
                log_probs = []
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    if self.config.vllm_log_prob and \
                        hasattr(output.outputs[sample_id], 'logprobs') and \
                        output.outputs[sample_id].logprobs is not None:
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
                if self.config.n > 1 and do_sample and not prompts.meta_info.get('agent_rollout', False):
                    single_idx = _repeat_interleave(single_idx, self.sampling_params.n)
                    single_attention_mask = _repeat_interleave(single_attention_mask, self.sampling_params.n)
                    single_position_ids = _repeat_interleave(single_position_ids, self.sampling_params.n)
                    # Create interleaved non_tensor_batch for this subset
                    for key, val in non_tensor_batch.items():
                        # Get single value and repeat n times
                        non_tensor_batch[key] = _repeat_interleave(val, self.sampling_params.n)

                seq = torch.cat([single_idx, response], dim=-1)
                
                response_length = response.size(1)
                delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).repeat(response.size(0), 1)
                if position_ids.dim() == 3:  # qwen2vl mrope
                    delta_position_id = delta_position_id.view(response.size(0), 1, -1).expand(response.size(0), 3, -1)
                
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
                
                
                if self.reward_fn is not None and not is_validate:
                    init_batch = DataProto(
                        batch=batch,
                        non_tensor_batch=non_tensor_batch,
                        meta_info=prompts.meta_info
                    )
                    reward_tensor = self.reward_fn(init_batch)
                    batch['token_level_scores'] = reward_tensor
                
                yield DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                    meta_info=prompts.meta_info
                )

        # Create new event loop for this generator
        for generated_seq in run_async_generator(_async_generate):
            yield generated_seq

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        return None


    @torch.no_grad()
    def generate_sequences_async_tool(self, prompts: DataProto, **kwargs):
        """Generator function that yields outputs as they complete (may be out of order).
        
        Args:
            prompts: DataProto containing the input prompts
            **kwargs: Additional arguments to modify sampling parameters
        
        Yields:
            tuple: (prompt_idx, DataProto) containing the original prompt index and its generated sequence
        """
        assert self.config.async_engine, "generate_sequences_async_tool requires `async_engine=True`"

        tool_stop_tokens = ["```\n\n"]
        tool_start_tokens = ["```python"]

        tool_response_start_tokens = ["```output\n"]
        tool_response_stop_tokens = ["\n```\n"]
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        
        if is_validate:
            # TODO: try **
            kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            })

        updated_sampling_params = deepcopy(self.sampling_params)
        for key, value in kwargs.items():
            if hasattr(updated_sampling_params, key):
                setattr(updated_sampling_params, key, value)
        # Helper function to call tools
        async def _apply_tool(tool_call, id=None):
            # if id is not None:
            #     tool_call["parameters"]["id"] = id
            # tool_call_result = await self.tool_caller(tool_call["name"], tool_call["parameters"])
            # return tool_call_result
            from rllm.rewards.code_utils.livecodebench import run_code
            tool_call_result = run_code(tool_call["parameters"]["code"])
            return {"content": tool_call_result}

        # Main function to process single prompt
        async def _generate_single_prompt_vectorized(prompt_idx: int, vllm_inputs: List[Dict[str, Any]]):
            """Process a single prompt with potential tool calls."""
            prompt_tokens = list(vllm_inputs[prompt_idx]['prompt_token_ids'])
            max_token_limit = self.config.response_length + len(prompt_tokens)
            max_response_token_limit = self.config.response_length 
       
            # Treat each generation indpendently by creating self.config.n independent generations.
            updated_local_sampling_params = deepcopy(updated_sampling_params)
            updated_local_sampling_params.stop = tool_stop_tokens
            updated_local_sampling_params.include_stop_str_in_output = True
            updated_local_sampling_params.n = 1
            updated_local_sampling_params.detokenize = True

            n_rollouts = self.config.n
            if is_validate:
                # validation data is already repeated in the outer loop
                n_rollouts = 1

            all_prompt_tokens: List[List[int]] = [copy.deepcopy(prompt_tokens) for _ in range(n_rollouts)]
            all_tokens = copy.deepcopy(all_prompt_tokens)    
            all_log_probs: List[List[float]] = [[] for _ in range(n_rollouts)]
            all_tool_call_masks: List[List[int]] = [[] for _ in range(n_rollouts)]
            all_dones = [False for _ in range(n_rollouts)]
            all_tool_call_counts = [0 for _ in range(n_rollouts)]
            has_tool_calls = [False for _ in range(n_rollouts)]  

            for _ in range(self.config.max_tool_calls):
                updated_local_sampling_params.max_tokens = max_token_limit - min([len(t) for idx,t in enumerate(all_tokens) if not all_dones[idx]])
                print(updated_local_sampling_params.max_tokens)
                gathered_outputs = await asyncio.gather(*[self.generate_sequence_task(idx, all_tokens[idx], updated_local_sampling_params) \
                                                          for idx in range(n_rollouts) if not all_dones[idx]])
        
                for gen_idx, gen_output in gathered_outputs:
                    output = gen_output.outputs[0]
                    cur_token_ids = list(output.token_ids)
                    if hasattr(output, 'logprobs') and output.logprobs is not None:
                        cur_logprob = [next(iter(log_prob.values())).logprob 
                                        for log_prob in output.logprobs]
                    else:
                        cur_logprob = [0.0] * len(output.token_ids)

                    cur_text = self.tokenizer.decode(cur_token_ids)
                    tool_calls = self.tool_caller.parse_tool_calls(cur_text)                    

                    if len(tool_calls) > 0:
                        tool_call = tool_calls[-1]
                        toolcall_result = await _apply_tool(tool_call)
                        if not isinstance(toolcall_result['content'], str):
                            # Timeout can return a Tuple
                            toolcall_result['content'] = str(toolcall_result['content'])
                        toolcall_result_with_special_token = "\n" + tool_response_start_tokens[0] + toolcall_result['content'] + tool_response_stop_tokens[0]
                        result_tokens = self.tokenizer.encode(toolcall_result_with_special_token, add_special_tokens=False)
                        result_tokens = list(result_tokens)
                        
                        # Update running statistics.
                        all_tokens[gen_idx].extend(cur_token_ids + result_tokens)
                        all_log_probs[gen_idx].extend(cur_logprob)
                        all_log_probs[gen_idx].extend([0.0] * (len(result_tokens)))
                        all_tool_call_masks[gen_idx].extend([1] * (len(cur_token_ids)))
                        all_tool_call_masks[gen_idx].extend([0] * (len(result_tokens)))
                        all_tool_call_counts[gen_idx] += 1
                        has_tool_calls[gen_idx] = True
                        if all_tool_call_counts[gen_idx] >= self.config.max_tool_calls:
                            all_dones[gen_idx] = True
                    else:
                        # Update running statistics.
                        all_tokens[gen_idx].extend(cur_token_ids)
                        all_log_probs[gen_idx].extend(cur_logprob)
                        all_tool_call_masks[gen_idx].extend([1] * (len(cur_token_ids)))
                        all_dones[gen_idx] = True

                    # If sequences has hit the max token limit, truncate the sequence and mark as done.
                    if len(all_tokens[gen_idx]) >= max_token_limit:
                        all_tokens[gen_idx] = all_tokens[gen_idx][:max_token_limit]
                        all_log_probs[gen_idx] = all_log_probs[gen_idx][:max_response_token_limit]
                        all_tool_call_masks[gen_idx] = all_tool_call_masks[gen_idx][:max_response_token_limit]
                        all_dones[gen_idx] = True
                # Break out of the loop if all generations have terminated.
                if all(all_dones):
                    break

            outputs = [
                SimpleNamespace(token_ids=tokens[len(prompt):], logprobs=probs, tool_call_mask=tool_call_mask, has_tool_call=has_tool_call)
                for tokens, probs, prompt, tool_call_mask, has_tool_call in zip(all_tokens, all_log_probs, all_prompt_tokens, all_tool_call_masks, has_tool_calls)
            ]
            final_output = SimpleNamespace(outputs=outputs)
            return prompt_idx, final_output

        async def _async_generate():
            # Create all tasks
            tasks = [
                _generate_single_prompt_vectorized(prompt_idx, vllm_inputs)
                for prompt_idx in range(batch_size)
            ]

            # Use as_completed to yield results as they finish
            for completed_task in asyncio.as_completed(tasks):
                prompt_idx, output = await completed_task
                # Finally, begin to post-process to return DataProto object.
                response = []
                log_probs = []
                tool_call_masks = []
                has_tool_calls = []
               
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    log_probs.append(output.outputs[sample_id].logprobs)
                    tool_call_masks.append(output.outputs[sample_id].tool_call_mask)
                    has_tool_calls.append(output.outputs[sample_id].has_tool_call)

                response = pad_2d_list_to_length(response, self.pad_token_id,
                                               max_length=self.config.response_length).to(idx.device)
                log_probs = pad_2d_list_to_length(log_probs, 0.0,
                                               max_length=self.config.response_length).to(idx.device)
                tool_call_masks = pad_2d_list_to_length(tool_call_masks, 0,
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
                        non_tensor_batch[key] = _repeat_interleave(val, self.sampling_params.n)

                # adding has_toolcall to the extra_info so that reward fn can assign bonus
                for i, info in enumerate(non_tensor_batch['extra_info']):
                    info['has_toolcall'] = has_tool_calls[i]


                seq = torch.cat([single_idx, response], dim=-1)
                
                response_length = response.size(1)
                delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).repeat(response.size(0), 1)
                if position_ids.dim() == 3:  # qwen2vl mrope
                    delta_position_id = delta_position_id.view(response.size(0), 1, -1).expand(response.size(0), 3, -1)
                
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
                        'position_ids': position_ids_out,
                        'tool_call_mask': tool_call_masks,
                    },
                    batch_size=response.size(0))

                if self.config.vllm_log_prob:
                    batch['old_log_probs'] = log_probs
                
                if self.reward_fn is not None and not is_validate:
                    init_batch = DataProto(
                        batch=batch,
                        non_tensor_batch=non_tensor_batch,
                        meta_info=prompts.meta_info
                    )
                    reward_tensor = self.reward_fn(init_batch)
                    batch['token_level_scores'] = reward_tensor
                
                yield DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                    meta_info=prompts.meta_info
                )

        # Create new event loop for this generator
        for generated_seq in run_async_generator(_async_generate):
            yield generated_seq

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        return None
