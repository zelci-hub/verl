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
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import json
try:
    import orjson  # Optional, faster JSON
    _HAS_ORJSON = True
except Exception:
    orjson = None  # type: ignore
    _HAS_ORJSON = False
import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any

# Problem ID context manager is available through ArcticInference plugin
# No need to import manually - vllm.plugins.load_general_plugins() handles this

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

# C++对象级锁定并行SuffixCache构建已集成到SuffixCache类中
# 不再需要全局multiprocessing函数

try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers import AutoConfig
    
    # Store original methods
    orig_mapping_register = CONFIG_MAPPING.register
    orig_auto_register = AutoConfig.register
    
    # Create safe wrappers that allow aimv2 duplicates
    def safe_mapping_register(key, value, exist_ok=False):
        if key == "aimv2":
            exist_ok = True
        return orig_mapping_register(key, value, exist_ok=exist_ok)
    
    def safe_auto_register(model_type, config, exist_ok=False):
        if model_type == "aimv2":
            exist_ok = True
        return orig_auto_register(model_type, config, exist_ok=exist_ok)
    
    # Apply patches
    CONFIG_MAPPING.register = safe_mapping_register
    AutoConfig.register = safe_auto_register
except:
    pass

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

# Import SuffixCache for speculative decoding
try:
    from arctic_inference.common.suffix_cache import SuffixCache
except ImportError:
    SuffixCache = None

import vllm
vllm.plugins.load_general_plugins()

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


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

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        
        # Initialize suffix cache data storage path if available
        self.suffix_cache_data_path = config.get("suffix_cache_data_path", None)
        # Optional: path to problem_id_to_files mapping (overrides default in suffix_cache_data_path)
        self.suffix_cache_mapping_path = config.get("suffix_cache_mapping_path", None)
        # Optional: window size (how many latest files per problem to read)
        try:
            self.suffix_cache_window_size = int(config.get("suffix_cache_window_size", 2))
        except Exception:
            self.suffix_cache_window_size = 2
        
        # Initialize speculative_config from engine_kwargs
        # Check if speculative_config was passed in engine_kwargs
        original_engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(config.engine_kwargs.vllm)
        )
        self.speculative_config = original_engine_kwargs.get("speculative_config", None)
        
        # Problem ID context manager is ready to use (no installation needed)

    def _load_suffix_cache_data_for_problem_ids(self, problem_ids):
        """
        Load suffix cache bootstrap data for specific problem IDs from previous epochs.
        
        Args:
            problem_ids (list): List of problem IDs to load cache data for
            
        Returns:
            dict: Mapping from problem_id to list of token sequences for suffix cache bootstrap
        """
        if not self.suffix_cache_data_path or not problem_ids:
            return {}
            
        problem_id_to_sequences = {}

        # Check if the path exists
        if not os.path.exists(self.suffix_cache_data_path):
            print(f"Suffix cache data path does not exist: {self.suffix_cache_data_path}")
            return {}
        # Load mapping of problem_id -> file list and select top-2 numerically largest files
        # Determine mapping path: prefer explicit setting, otherwise default inside suffix_cache_data_path
        mapping_path = self.suffix_cache_mapping_path or (
            os.path.join(self.suffix_cache_data_path, "problem_id_to_files.json")
            if self.suffix_cache_data_path else None
        )
        if not mapping_path:
            print("No mapping_path available (suffix_cache_mapping_path and suffix_cache_data_path are empty)")
            return {}
        if not os.path.exists(mapping_path):
            print(f"problem_id_to_files mapping not found: {mapping_path}")
            return {}
        try:
            with open(mapping_path, "r", encoding="utf-8") as mf:
                pid_to_files_map = json.load(mf)
        except Exception as e:
            print(f"Failed to load mapping file: {e}")
            return {}

        base_dir = os.path.dirname(mapping_path)
        pid_str_to_original = {str(pid): pid for pid in problem_ids}
        file_to_needed_pids = {}

        for pid_str, orig_pid in pid_str_to_original.items():
            filenames = pid_to_files_map.get(pid_str, [])
            if not filenames:
                continue

            def parse_num(name):
                try:
                    return int(os.path.splitext(name)[0])
                except Exception:
                    return -1

            filenames_sorted = sorted(filenames, key=parse_num)
            window_size = self.suffix_cache_window_size if self.suffix_cache_window_size and self.suffix_cache_window_size > 0 else 2
            selected = filenames_sorted[-window_size:] if len(filenames_sorted) >= window_size else filenames_sorted
            for fname in selected:
                file_path = os.path.join(base_dir, fname)
                file_to_needed_pids.setdefault(file_path, set()).add(pid_str)

        if not file_to_needed_pids:
            print("No files selected from mapping for requested problem_ids.")
            return {}

        processed_files = 0
        try:
            # Prepare global early-stop tracking when a per-pid maximum is configured
            max_per_pid = getattr(self, "suffix_cache_max_sequences_per_pid", None)
            if max_per_pid is not None:
                all_needed_pid_strs = set()
                for s in file_to_needed_pids.values():
                    all_needed_pid_strs.update(s)
                collected_per_pid = {pid_str: 0 for pid_str in all_needed_pid_strs}
            else:
                collected_per_pid = None

            for file_path, needed_pid_strs in file_to_needed_pids.items():
                processed_files += 1
                print(f"Loading suffix cache data from: {file_path}")
                # Work on a per-file mutable copy for early termination within the file
                pending_pid_strs = set(needed_pid_strs)
                try:
                    # Use large buffering and binary mode; prefer orjson when available
                    with open(file_path, 'rb', buffering=1 << 20) as f:
                        for line in f:
                            # Fast pre-filter: skip lines that cannot contain any target pid
                            if pending_pid_strs and not any(pid.encode() in line for pid in pending_pid_strs):
                                continue

                            # Parse JSON (prefer orjson on bytes; fallback to json on str)
                            try:
                                if _HAS_ORJSON:
                                    data = orjson.loads(line)
                                else:
                                    data = json.loads(line.decode('utf-8').strip())
                            except Exception:
                                continue

                            pid_value = data.get('problem_id')
                            if pid_value is None or 'output_token_ids' not in data:
                                continue
                            pid_as_str = str(pid_value)
                            if pid_as_str not in pending_pid_strs:
                                continue

                            orig_pid = pid_str_to_original.get(pid_as_str, pid_value)
                            if orig_pid not in problem_id_to_sequences:
                                problem_id_to_sequences[orig_pid] = []
                            problem_id_to_sequences[orig_pid].append(data['output_token_ids'])

                            # Update early-stop tracking
                            if collected_per_pid is not None:
                                collected_per_pid[pid_as_str] += 1
                                if collected_per_pid[pid_as_str] >= max_per_pid:
                                    pending_pid_strs.discard(pid_as_str)
                                    # If this file no longer needs any pids, stop reading it
                                    if not pending_pid_strs:
                                        break
                except FileNotFoundError:
                    continue

                # Global early stop: if we satisfied all pids across files, end the loop
                if collected_per_pid is not None and all(v >= max_per_pid for v in collected_per_pid.values()):
                    break
        except Exception as e:
            print(f"Failed to load suffix cache data: {e}")
            
        print(f"Loaded suffix cache data for {len(problem_id_to_sequences)} problem IDs from {processed_files} files")
          
        return problem_id_to_sequences

    def get_prompt_token_ids(self, vllm_inputs, problem_id):
        """
        Get prompt token IDs for a specific problem_id from vllm_inputs.
        
        Args:
            vllm_inputs (list): List of vllm input dictionaries, each containing:
                - 'prompt_token_ids': List of token IDs for the prompt
                - 'problem_id': Problem identifier
                - 'multi_modal_data': Optional multi-modal data
            problem_id: The problem ID to search for
        
        Returns:
            list[int] or None: The prompt token IDs for the matching problem_id, or None if not found
        """
        for vllm_input in vllm_inputs:
            if vllm_input.get("problem_id") == problem_id:
                return vllm_input.get("prompt_token_ids")

        print(f"DEBUG: No prompt token IDs found for problem_id: {problem_id}")
        return None

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

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # Extract problem_id if available
        problem_ids = non_tensor_batch.get("problem_id", None)
        # print("DEBUG:in generate_sequences: problem_ids", problem_ids)
        
        # Handle problem_id - generate if not present
        if "problem_id" in non_tensor_batch:
            problem_ids = non_tensor_batch.pop("problem_id")
        else:
            # Generate problem_ids if not present in the data
            batch_size = len(non_tensor_batch["raw_prompt_ids"])
            problem_ids = [f"generated_{i:05d}" for i in range(batch_size)]
            print(f"WARNING: problem_id not found in batch, generated {batch_size} problem_ids")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for i, (raw_prompt_ids, multi_modal_data, problem_id) in enumerate(zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), problem_ids,
                strict=True
            )):
                vllm_input = {"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data, "problem_id": problem_id}
                vllm_inputs.append(vllm_input)
        else:
            vllm_inputs = []
            for i, (raw_prompt_ids, problem_id) in enumerate(zip(non_tensor_batch.pop("raw_prompt_ids"), problem_ids)):
                vllm_input = {"prompt_token_ids": raw_prompt_ids, "problem_id": problem_id}
                vllm_inputs.append(vllm_input)

        #print("DEBUG: sample problem_ids:", [x["problem_id"] for x in vllm_inputs])

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size


        if (self.speculative_config is not None and
            self.speculative_config.get("enable_suffix_decoding", False)):
            if self.speculative_config.get("method") not in (
                    "arctic", "suffix", "mlp_speculator"):
                raise ValueError(
                    "Suffix decoding is only supported with the 'arctic', "
                    "'mlp_speculator' or 'suffix' spec decoding methods.")
            if SuffixCache is None:
                raise ImportError("SuffixCache not available. Please install arctic_inference package.")
            suffix_cache_max_depth = self.speculative_config.get("suffix_cache_max_depth", 64)
            suffix_cache_max_threads = self.speculative_config.get("suffix_cache_max_threads", None)
            
            # 🎯 SuffixCache线程数配置：硬编码为10线程
            if suffix_cache_max_threads is None:
                suffix_cache_max_threads = 10
                print(f"🎯 SuffixCache使用硬编码线程数: {suffix_cache_max_threads}")
            else:
                print(f"🎯 SuffixCache使用配置指定线程数: {suffix_cache_max_threads}")
            
            # 🚀 启用C++对象级锁定+GIL释放的线程安全模式
            self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache = SuffixCache(
                max_depth=suffix_cache_max_depth, 
                thread_safe=True, 
                max_threads=suffix_cache_max_threads
            )            
            
            enable_suffix_prebuild = self.suffix_cache_data_path is not None

            if enable_suffix_prebuild:
                assert problem_ids is not None, "problem_ids must be provided when enable_suffix_prebuild is True"
                unique_problem_ids = list(set(problem_ids))   
                import time
                total_start_time = time.perf_counter()       
                problem_id_to_sequences = self._load_suffix_cache_data_for_problem_ids(unique_problem_ids)
                total_time = time.perf_counter() - total_start_time
                print(f"DEBUG: 数据加载总耗时: {total_time:.4f}秒，处理了{len(unique_problem_ids)}个problem_ids")

                assert problem_id_to_sequences is not None, "problem_id_to_sequences must be provided when enable_suffix_prebuild is True"
                #print(f"🚀 SuffixCache C++对象级锁定并行构建: {len(problem_id_to_sequences)} 个问题")
                
                # 获取suffix_cache引用（现在已经是线程安全模式）
                suffix_cache = self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache
                
                # 🔥 准备并行批处理数据格式
                problems_data = []
                for problem_id in unique_problem_ids:
                    if problem_id in problem_id_to_sequences:
                        prompt_tokens = self.get_prompt_token_ids(vllm_inputs, problem_id)
                        if prompt_tokens is not None:
                            sequences = problem_id_to_sequences[problem_id]
                            problems_data.append((problem_id, prompt_tokens, sequences))
                
                #print(f"📊 准备并行处理: {len(problems_data)} 个问题, "f"总序列数: {sum(len(seqs) for _, _, seqs in problems_data)}")
                
                # ⚡ 使用C++对象级锁定+GIL释放的真正并行处理
                import time
                total_start_time = time.perf_counter()               
                parallel_result = suffix_cache.prebuild_problems_parallel(problems_data)
                total_time = time.perf_counter() - total_start_time

                # 📈 性能报告（健壮性：缺失键时使用回退值）
                successful = parallel_result.get("successful_problems")
                if successful is None:
                    successful = parallel_result.get("total_problems", 0)
                active_threads = parallel_result.get("active_threads", "N/A")
                actual_speedup = parallel_result.get("actual_speedup", "N/A")

                # print(f"🎯 C++对象级锁定并行构建完成:")
                # print(f"  ✅ 成功问题: {successful}/{len(problems_data)}")
                # print(f"  ⚡ 总时间: {total_time:.4f}秒")
                # print(f"  🚀 实际加速: {actual_speedup}x")
                # print(f"  🧵 活跃线程: {active_threads}")
                # print(f"  🔒 技术: 每个SuffixTree独立C++锁+GIL释放")
                
                # 验证结果
                cache_stats = suffix_cache.get_cache_stats()
                print(f"  📊 最终统计: {cache_stats['problem_tree_count']} 个问题树, "
                      f"{cache_stats['total_sequences']} 个序列")

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # Initialize context manager for problem_id to req_id mapping if problem_ids provided
            try:
                # Import ArcticInference plugin's ProblemIdContextManager
                from arctic_inference.vllm.model_runner import ProblemIdContextManager
                
                # Create empty req_id to problem_id mapping context
                ProblemIdContextManager.clear_req_id_mapping()
                ProblemIdContextManager.set_req_id_to_problem_id_mapping({})
                
                # Call generate with problem_ids parameter - LLM patches will handle the mapping
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                    problem_ids=problem_ids,  # Pass problem_ids to generate method
                )

            except (ImportError, TypeError):
                # Fallback if ArcticInference plugin is not available or LLM patches are disabled
                print("Warning: ArcticInference LLM plugin not available or disabled, problem_ids will be ignored")
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )
                
            # Clear suffix cache after generation to free up C++ memory
            try:    
                self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache.clear_all_cache()
            except Exception as e:
                print(f"DEBUG: Failed to clear suffix cache: {e}")

            # Output generation length statistics
            total_generated_tokens = 0
            generation_lengths = []
            for i, output in enumerate(outputs):
                for sample_id in range(len(output.outputs)):
                    generated_length = len(output.outputs[sample_id].token_ids)
                    generation_lengths.append(generated_length)
                    total_generated_tokens += generated_length
            
            # Get rank information
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            else:
                rank = int(os.environ.get("RANK", "0"))
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            
            if generation_lengths:
                avg_length = total_generated_tokens / len(generation_lengths)
                min_length = min(generation_lengths)
                max_length = max(generation_lengths)
                print(f"[Rank {rank}/Local {local_rank}] Generation Length Stats: Total={total_generated_tokens}, "
                      f"Count={len(generation_lengths)}, Avg={avg_length:.2f}, "
                      f"Min={min_length}, Max={max_length}")
                print(f"[Rank {rank}/Local {local_rank}] Individual lengths: {generation_lengths[:10]}..." if len(generation_lengths) > 10 else f"[Rank {rank}/Local {local_rank}] Individual lengths: {generation_lengths}")

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

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
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
