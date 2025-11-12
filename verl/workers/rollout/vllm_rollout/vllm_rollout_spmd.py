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
import time
import queue
import datetime
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any
import concurrent.futures as _futures

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

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.single_controller.base.decorator import register
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

# Import SuffixDecodingCache for speculative decoding
try:
    from arctic_inference.suffix_decoding.cache import SuffixDecodingCache
except Exception as e:
    raise ImportError(f"Failed to import Arctic-Inference: {e}")

import vllm
vllm.plugins.load_general_plugins()

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "DEBUG"))  # 临时改为DEBUG级别

def _log_performance_info(event_name: str, extra_info: str = ""):
    """Log performance monitoring information with timestamp, thread, and process info"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # millisecond precision
    thread_id = threading.current_thread().ident
    thread_name = threading.current_thread().name
    process_id = os.getpid()
    
    # Try to get CPU usage if available
    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        extra_info += f" | CPU: {cpu_percent:.1f}% | Memory: {memory_info.percent:.1f}%"
    except ImportError:
        pass
    
    logger.info(f"[PERF] {timestamp} | PID:{process_id} | Thread:{thread_name}({thread_id}) | {event_name} {extra_info}")

# ---------------------------------------------------------------------------
# Module-level helper for parallel suffix cache scanning
# ---------------------------------------------------------------------------
def _suffix_cache_scan_file(
    file_path: str,
    needed_pid_strs: list[str],
    pid_str_to_original: dict[str, Any],
    max_per_pid: int | None,
    has_orjson: bool,
):
    """Scan a single JSONL file and collect sequences for target problem IDs.

    Returns a tuple: (orig_pid_to_sequences, per_pid_counts)
    - orig_pid_to_sequences: dict[orig_pid, list[list[int]]]
    - per_pid_counts: dict[pid_str, int]
    """
    results: dict[Any, list] = {}
    per_pid_counts: dict[str, int] = {pid: 0 for pid in needed_pid_strs}

    pending_pid_strs = set(needed_pid_strs)

    try:
        with open(file_path, 'rb', buffering=1 << 20) as f:
            load_json = orjson.loads if has_orjson else None
            pid_bytes = {pid: pid.encode() for pid in pending_pid_strs}
            max_per = max_per_pid

            for line in f:
                if pending_pid_strs and not any(pb in line for pb in pid_bytes.values()):
                    continue
                try:
                    if load_json is not None:
                        data = load_json(line)
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
                if orig_pid not in results:
                    results[orig_pid] = []
                results[orig_pid].append(data['output_token_ids'])

                if max_per is not None:
                    per_pid_counts[pid_as_str] += 1
                    if per_pid_counts[pid_as_str] >= max_per:
                        pending_pid_strs.discard(pid_as_str)
                        if not pending_pid_strs:
                            break
    except FileNotFoundError:
        return {}, per_pid_counts
    except Exception:
        return results, per_pid_counts

    return results, per_pid_counts

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
        self.trainer_rollout_data_dir = kwargs.pop("trainer_rollout_data_dir", None)  # Get trainer rollout_data_dir

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
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        
        # Initialize speculative_config from engine_kwargs
        # Check if speculative_config was passed in engine_kwargs
        original_engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(config.engine_kwargs.vllm)
        )
        self.speculative_config = original_engine_kwargs.get("speculative_config", None)
        
        # Initialize suffix cache configuration
        self.enable_suffix_prebuild = config.get("enable_suffix_prebuild", False)
        if self.enable_suffix_prebuild:
            self.suffix_cache_data_path = self.trainer_rollout_data_dir
            assert self.suffix_cache_data_path is not None, "rollout_data_dir must be set when enable_suffix_prebuild is True"
            if config.get("suffix_cache_window_size") is not None:
                self.suffix_cache_window_size = int(config.get("suffix_cache_window_size"))
            else:
                self.suffix_cache_window_size = 4 
                logger.info(f"suffix_cache_window_size not set, using default value: {self.suffix_cache_window_size}")
            
            # Initialize problem_id_to_files mapping window from file_to_problem_ids.jsonl
            self._problem_id_to_files_window = {}
            self._mapping_loaded = False  # Flag to track if mapping has been loaded
            
            # Initialize suffix prebuild infrastructure (background queue + worker thread)
            self._suffix_prebuild_queue: queue.Queue = queue.Queue()
            # Track ongoing prebuild tasks for synchronization
            self._active_prebuild_pids = set()  # Currently being processed problem IDs
            self._prebuild_lock = threading.Lock()  # Lock for thread-safe access to active_pids
        # Cache of length-rank categories computed during prebuild
            self._hard_pid_set = set()
            self._medium_pid_set = set()
            self._easy_pid_set = set()
            self._suffix_prebuild_thread = threading.Thread(
                target=self._suffix_prebuild_loop, name="suffix-prebuild-worker", daemon=True
            )
            self._suffix_prebuild_thread.start()
        else:
            # Initialize default values when suffix prebuild is disabled
            self.suffix_cache_data_path = None
            self.suffix_cache_window_size = 2
            self._problem_id_to_files_window = {}
            self._mapping_loaded = False  # Flag to track if mapping has been loaded
            self._suffix_prebuild_queue = None
            self._active_prebuild_pids = set()
            self._prebuild_lock = threading.Lock()
            self._hard_pid_set = set()
            self._medium_pid_set = set()
            self._easy_pid_set = set()
            self._suffix_prebuild_thread = None

    def _load_problem_id_mapping_from_file(self, max_iteration=None):
        """
        Load problem_id_to_files mapping from file_to_problem_ids.jsonl and build local window.
        This method reads from the suffix_cache_data_path and constructs the mapping window based on suffix_cache_window_size.
        
        Args:
            max_iteration: Optional maximum iteration to filter files. Only files with step <= max_iteration will be included.
        """
        if not self.enable_suffix_prebuild:
            return
            
        # Assert that interleave is False for suffix cache functionality
        assert not self.config.get("interleave", True), "now prebuild Suffix cache requires interleave=False"
            
        # Look for file_to_problem_ids.jsonl in the suffix_cache_data_path
        file_to_problem_ids_path = os.path.join(self.suffix_cache_data_path, "file_to_problem_ids.jsonl")
        
        if not os.path.exists(file_to_problem_ids_path):
            logger.warning(f"file_to_problem_ids.jsonl not found at {file_to_problem_ids_path}")
            return
            
        logger.info(f"Loading problem_id mapping from {file_to_problem_ids_path}")
        
        try:
            # Read all entries from file_to_problem_ids.jsonl
            file_entries = []
            with open(file_to_problem_ids_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if 'filename' in entry and 'problem_ids' in entry and 'step' in entry and entry['step'] <= max_iteration:
                            file_entries.append(entry)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in {file_to_problem_ids_path} line {line_num}: {e}")
                        continue
            
            if not file_entries:
                logger.warning("No valid entries found in file_to_problem_ids.jsonl")
                return
                
            # Build problem_id_to_files mapping with window size limit
            problem_id_to_all_files = {}
            for entry in file_entries:
                filename = entry['filename']
                problem_ids = entry['problem_ids']
                step = entry.get('step', None)
                if step is None:
                    logger.warning(f"step is not found for filename: {filename}")
                    continue
                
                for problem_id in problem_ids:
                    if problem_id not in problem_id_to_all_files:
                        problem_id_to_all_files[problem_id] = []
                    problem_id_to_all_files[problem_id].append({"filename": filename, "step": step})
            
            # Apply max_iteration filter and window size limit - keep only the latest files for each problem_id
            for problem_id, file_list in problem_id_to_all_files.items():
                # Sort by step (descending) and keep only the latest window_size files
                file_list.sort(key=lambda x: x["step"], reverse=True)
                latest_files = file_list[:self.suffix_cache_window_size]
                self._problem_id_to_files_window[problem_id] = [f for f in latest_files]
            
            filter_info = f" (max_iteration={max_iteration})" if max_iteration is not None else ""
            logger.info(f"Loaded mapping for {len(self._problem_id_to_files_window)} problem_ids with window_size={self.suffix_cache_window_size}{filter_info}")
            
            # Log some statistics
            local_rank = os.getenv("LOCAL_RANK", "0")
            total_files = sum(len(files) for files in self._problem_id_to_files_window.values())
            avg_files_per_problem = total_files / len(self._problem_id_to_files_window) if self._problem_id_to_files_window else 0
            if local_rank == 0:
                logger.info(f"Average files per problem_id: {avg_files_per_problem:.2f}")
            
            
        except Exception as e:
            logger.error(f"Failed to load problem_id mapping from file: {e}")
            self._problem_id_to_files_window = {}

    def _load_suffix_cache_data_for_problem_ids(self, problem_ids):
        """
        Load suffix cache bootstrap data for specific problem IDs from previous epochs.
        
        Args:
            problem_ids (list): List of problem IDs to load cache data for
            
        Returns:
            dict: Mapping from problem_id to list of token sequences for suffix cache bootstrap
        """
                # Get process info for filename
        local_rank = os.getenv("LOCAL_RANK", "0")
        rank = os.getenv("RANK", "0")  
        if not self.enable_suffix_prebuild or not problem_ids:
            return {}
            
        problem_id_to_sequences = {}

        # Check if the data path exists
        assert os.path.exists(self.suffix_cache_data_path), f"suffix cache data path does not exist: {self.suffix_cache_data_path}"
        
        # Use local problem_id_to_files mapping window instead of reading from file
        base_dir = self.suffix_cache_data_path
        pid_str_to_original = {str(pid): pid for pid in problem_ids}
        file_to_needed_pids = {}
        
        # Track file selection info for each problem_id
        pid_to_selected_files = {}

        def parse_num(name):
            try:
                return int(os.path.splitext(name)[0])
            except Exception:
                return -1

        for pid_str, orig_pid in pid_str_to_original.items():
            # Get file info from local mapping window
            file_infos = self._problem_id_to_files_window.get(int(pid_str), []) if pid_str.isdigit() else []
            if not file_infos:
                logger.warning(f"No file info found for problem_id: {pid_str}")
                continue

            # Extract filenames from file info (handle both old format and new format)
            if file_infos and isinstance(file_infos[0], dict):
                # New format: list of {"filename": "xxx.jsonl", "step": 123}
                filenames = [info["filename"] for info in file_infos]
            else:
                # Old format: list of filenames
                filenames = file_infos
            
            # Record selected files for this problem_id
            pid_to_selected_files[pid_str] = filenames
            
            for fname in filenames:
                file_path = os.path.join(base_dir, fname)
                file_to_needed_pids.setdefault(file_path, set()).add(pid_str)

        if not file_to_needed_pids:
            logger.warning("No files selected from mapping for requested problem_ids.")
            return {}
        
        # Print file range statistics
        all_selected_files = []
        for files in pid_to_selected_files.values():
            all_selected_files.extend(files)
        if all_selected_files:
            def parse_num(name):
                try:
                    return int(os.path.splitext(name)[0])
                except Exception:
                    return -1
            file_numbers = [parse_num(f) for f in all_selected_files if parse_num(f) != -1]
            if file_numbers:
                min_file_num = min(file_numbers)
                max_file_num = max(file_numbers)
                if rank == 0:
                    logger.info(f"File range: [{min_file_num}, {max_file_num}], window_size={self.suffix_cache_window_size}")
        
        # Print per-problem_id file selection (sample)
        if rank == 0:
            sample_pids = list(pid_to_selected_files.keys())[:1]
            for pid in sample_pids:
                files = pid_to_selected_files[pid]
                logger.info(f" sample problem_id={pid}: using {len(files)} files {files}")

        processed_files = 0
        try:
            # Prepare job list and global early-stop tracking
            max_per_pid = getattr(self, "suffix_cache_max_sequences_per_pid", None)
            jobs = []
            all_needed_pid_strs = set()
            for file_path, needed_pid_strs in file_to_needed_pids.items():
                jobs.append((file_path, list(needed_pid_strs)))
                all_needed_pid_strs.update(needed_pid_strs)

            max_workers = 10
            per_pid_counts_total: dict[str, int] = {pid: 0 for pid in all_needed_pid_strs}
            has_orjson = _HAS_ORJSON

            with _futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        _suffix_cache_scan_file,
                        fp,
                        pids,
                        pid_str_to_original,
                        max_per_pid,
                        has_orjson,
                    )
                    for (fp, pids) in jobs
                ]

                for fut in _futures.as_completed(futures):
                    processed_files += 1
                    try:
                        res_map, res_counts = fut.result()
                    except Exception:
                        continue

                    for k, v in res_map.items():
                        if k not in problem_id_to_sequences:
                            problem_id_to_sequences[k] = []
                        problem_id_to_sequences[k].extend(v)

                    for pid_str, cnt in res_counts.items():
                        per_pid_counts_total[pid_str] = per_pid_counts_total.get(pid_str, 0) + cnt

                    if max_per_pid is not None and all(c >= max_per_pid for c in per_pid_counts_total.values()):
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        break
        except Exception as e:
            logger.error(f"Failed to load suffix cache data: {e}")          
        # Print statistics: sequences count per problem_id
        if problem_id_to_sequences and local_rank == 0:
            sequence_counts = {pid: len(seqs) for pid, seqs in problem_id_to_sequences.items()}
            total_sequences = sum(sequence_counts.values())
            avg_sequences = total_sequences / len(sequence_counts) if sequence_counts else 0
            min_count = min(sequence_counts.values()) if sequence_counts else 0
            max_count = max(sequence_counts.values()) if sequence_counts else 0
            
            logger.info(f"📊 Sequence statistics: TOTAL={total_sequences}, AVG={avg_sequences:.1f}, MIN={min_count}, MAX={max_count}")
            logger.info(f"📊 Per-problem sequences: {len(sequence_counts)} problems loaded")
             
          
        return problem_id_to_sequences

    def _update_problem_id_to_files_mapping(self, filename, current_step,problem_ids):
        """Update problem_id_to_files mapping in local window with window size limit.
        
        Args:
            dump_path: Directory path where files are dumped (for compatibility)
            filename: Filename to add to the mapping
            problem_ids: List of problem_ids in this file
        """
        if not self.enable_suffix_prebuild:
            return
            
        # Assert that interleave is False for suffix cache functionality
        assert not self.config.get("interleave", True), "Suffix cache requires interleave=False"
        
        
        # Update mapping for each problem_id in local window
        for problem_id in set(problem_ids):
            if problem_id not in self._problem_id_to_files_window:
                self._problem_id_to_files_window[problem_id] = []
            
            # Add current file with step info
            file_info = {"filename": filename, "step": current_step}
            
            # Remove if already exists (to avoid duplicates)
            self._problem_id_to_files_window[problem_id] = [
                f for f in self._problem_id_to_files_window[problem_id] 
                if f.get("filename") != filename
            ]
            
            # Add new file
            self._problem_id_to_files_window[problem_id].append(file_info)
            
            # Sort by step (descending) and keep only the latest window_size files
            self._problem_id_to_files_window[problem_id].sort(key=lambda x: x["step"], reverse=True)
            self._problem_id_to_files_window[problem_id] = self._problem_id_to_files_window[problem_id][:self.suffix_cache_window_size]
        
        logger.debug(f"Updated local mapping window for {len(set(problem_ids))} problem_ids")


    def get_length_rank(self, problem_id_to_sequences):
        """Classify problems by mean and max output token length.

        This computes the mean and maximum sequence length (over all sequences) 
        for each ``problem_id`` and classifies them into three categories based 
        on sorted order:
        - hard: top 20% (sorted by mean desc, then max desc)
        - medium: next 30%
        - easy: next 30%

        Args:
            problem_id_to_sequences: dict mapping problem_id -> list of
                sequences (each sequence is a list of token IDs).

        Returns:
            tuple[list, list, list]: (hard_ids, medium_ids, easy_ids) where 
            each list contains problem_ids sorted by their (mean, max) length 
            in descending order.
        """
        lengths = {}
        for pid, sequences in problem_id_to_sequences.items():
            mean_len = 0
            max_len = 0
            if sequences:
                seq_len = [len(seq) for seq in sequences]
                mean_len = np.mean(seq_len)
                max_len = max(seq_len)
            lengths[pid] = (float(mean_len), float(max_len))
        
        # Sort by mean (descending), then by max (descending)
        sorted_pids = sorted(lengths.items(), 
                            key=lambda x: (x[1][0], x[1][1]), 
                            reverse=True)
        
        # Calculate split indices
        total_count = len(sorted_pids)
        hard_end = int(total_count * 0.20)
        medium_end = hard_end + int(total_count * 0.30)
        easy_end = medium_end + int(total_count * 0.30)
        
        # Split into categories
        hard_ids = [pid for pid, _ in sorted_pids[:hard_end]]
        medium_ids = [pid for pid, _ in sorted_pids[hard_end:medium_end]]
        easy_ids = [pid for pid, _ in sorted_pids[medium_end:easy_end]]
        rank = os.getenv("RANK", "0")
        if rank == 0:
            print(f"hard_mean{sorted_pids[:hard_end][0][1][0]}, hard_max{sorted_pids[:hard_end][0][1][1]}")
            print(f"medium_mean{sorted_pids[hard_end:medium_end][0][1][0]}, medium_max{sorted_pids[hard_end:medium_end][0][1][1]}")
            print(f"easy_mean{sorted_pids[medium_end:easy_end][0][1][0]}, easy_max{sorted_pids[medium_end:easy_end][0][1][1]}")
            
        return hard_ids, medium_ids, easy_ids

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

        logger.debug(f"DEBUG: No prompt token IDs found for problem_id: {problem_id}")
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

    def enqueue_prebuild(self, problem_ids, raw_prompt_ids=None, iteration: int | None = None):
        """Non-blocking: enqueue a prebuild task for given problem_ids.

        Args:
            problem_ids: list of problem IDs
            raw_prompt_ids: optional list of prompt token lists aligned with problem_ids
        """
        try:
            # Skip entirely if no suffix cache data path configured
            if not self.suffix_cache_data_path:
                return
            if not problem_ids:
                return
            task = {"problem_ids": list(problem_ids), "raw_prompt_ids": raw_prompt_ids, "iteration": iteration}
            self._suffix_prebuild_queue.put_nowait(task)
            # logger.info(f"📥 Enqueued prebuild task: {len(problem_ids)} problem_ids, iteration={iteration}")
            # logger.info(f"📥 Problem_ids preview: {list(problem_ids)[:20]}{'...' if len(problem_ids) > 20 else ''}")
            _log_performance_info("PREBUILD_ENQUEUED", f"pids={len(problem_ids)}, iter={iteration}")
        except Exception as e:
            logger.error(f"enqueue_prebuild failed: {e}")

    def _suffix_prebuild_loop(self):
        local_rank = os.getenv("LOCAL_RANK", "0")
        rank = os.getenv("RANK", "0") 
        while True:
            task = self._suffix_prebuild_queue.get()
            try:
                assert self.enable_suffix_prebuild, "suffix prebuild should be enabled"
                problem_ids = task.get("problem_ids") or None
                raw_prompt_ids = task.get("raw_prompt_ids") or None 
                max_iteration = task.get("iteration") or None
                assert problem_ids is not None, "problem_ids must be provided when enable_suffix_prebuild is True"
                assert raw_prompt_ids is not None, "raw_prompt_ids must be provided when enable_suffix_prebuild is True"
                assert max_iteration is not None, "max_iteration must be provided when enable_suffix_prebuild is True"
                # logger.info(f"🚀 PREBUILD STARTED: Processing {len(problem_ids)} problem_ids from queue (iteration={max_iteration})")
                # logger.info(f"🚀 Problem_ids: {problem_ids[:20]}{'...' if len(problem_ids) > 20 else ''}")
                _log_performance_info("PREBUILD_STARTED", f"pids={len(problem_ids)}, iter={max_iteration}")

                # Mark these problem IDs as being processed
                with self._prebuild_lock:
                    self._active_prebuild_pids.update(pid for pid in problem_ids)
                
                # DEBUG: Log received problem_ids
                unique_pids_received = sorted(list(set(problem_ids)))
                logger.info(f"🔍 PREBUILD DEBUG: Received {len(problem_ids)} problem_ids ({len(unique_pids_received)} unique)")

                # ensure suffix cache exists (thread-safe)
                try:
                    suffix_cache = self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache
                except Exception:
                    suffix_cache = None
                if suffix_cache is None:
                    if isinstance(self.speculative_config, dict):
                        max_depth = self.speculative_config.get("suffix_cache_max_depth", 64)
                        max_threads = self.speculative_config.get("suffix_cache_max_threads", 10)
                    suffix_cache = SuffixDecodingCache(max_tree_depth=max_depth, thread_safe=True, max_threads=max_threads)
                    self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache = suffix_cache

                # assemble problems_data
                unique_ids = list(dict.fromkeys([p for p in problem_ids]))
                pid_native = unique_ids
                try:
                    problem_id_to_sequences = self._load_suffix_cache_data_for_problem_ids(pid_native)
                except Exception as e:
                    logger.error(f"prebuild: load data failed: {e}")
                    problem_id_to_sequences = {}
                
                # DEBUG: Check which problem_ids have historical data
                pids_with_data = set(str(p) for p in problem_id_to_sequences.keys())
                pids_without_data = set(str(p) for p in unique_ids) - pids_with_data
                logger.info(f"🔍 PREBUILD DEBUG: {len(pids_with_data)} pids have historical data, {len(pids_without_data)} don't")

                # Select only hard and medium problems for prebuild
                try:
                    # Check if distribution_aware is enabled
                    distribution_aware = False
                    if isinstance(self.speculative_config, dict):
                        distribution_aware = self.speculative_config.get("distribution_aware", False)
                    
                    if distribution_aware and problem_id_to_sequences:
                        # distribution_aware=True: use length-based classification
                        hard_ids, medium_ids,easy_ids = self.get_length_rank(problem_id_to_sequences)
                        logger.info(f"🎯 Prebuild classification (distribution_aware=True): {len(hard_ids)} hard + {len(medium_ids)} medium problems (from {len(problem_id_to_sequences)} with historical data)")
                    else:
                        # distribution_aware=False or no historical data: treat all as medium
                        hard_ids, medium_ids,easy_ids = unique_ids, [], []
                        if distribution_aware:
                            logger.info(f"🎯 Prebuild (distribution_aware=True): no historical data, treating all {len(unique_ids)} problems as hard difficulty")
                        else:
                            logger.info(f"🎯 Prebuild (distribution_aware=False): treating all {len(unique_ids)} problems as hard difficulty")

                    allowed_pid_strs = set([str(p) for p in list(hard_ids) + list(medium_ids) + list(easy_ids)])
                    # Record categories for later reuse in generate_sequences
                    # Clear previous categories and update with current batch only
                    self._hard_pid_set.clear()
                    self._medium_pid_set.clear()
                    self._easy_pid_set.clear()
                    self._hard_pid_set.update(str(p) for p in hard_ids)
                    self._medium_pid_set.update(str(p) for p in medium_ids)
                    self._easy_pid_set.update(str(p) for p in easy_ids)
                    logger.info(f"🎯 Total allowed problem_ids for prebuild: {len(allowed_pid_strs)}")
                except Exception:
                    allowed_pid_strs = None

                pid_to_prompt = {}
                if raw_prompt_ids is not None:
                    for pid, pr in zip(problem_ids, raw_prompt_ids):
                        pid_to_prompt.setdefault(pid, pr)

                problems_data = []
                for pid in problem_ids:
                    # Filter by hard/medium selection if available
                    if allowed_pid_strs is not None and str(pid) not in allowed_pid_strs:
                        continue
                    
                    raw_sequences = problem_id_to_sequences.get(pid, problem_id_to_sequences.get(str(pid), []))
                    prompt_tokens = pid_to_prompt.get(pid)
                    
                    if raw_sequences and prompt_tokens is not None:
                        # Convert raw token sequences to the format expected by cache.py
                        formatted_sequences = []
                        for seq_idx, response_tokens in enumerate(raw_sequences):
                            formatted_sequences.append({
                                'seq_id': seq_idx,
                                'prompt_tokens': prompt_tokens,
                                'response_tokens': response_tokens
                            })
                        
                        problems_data.append({
                            'problem_id': pid,
                            'sequences': formatted_sequences
                        })
                    else:
                        # No historical sequences or no prompt tokens - create minimal prebuild data
                        # Use empty sequences list, SuffixCache will handle this gracefully
                        problems_data.append({
                            'problem_id': pid,
                            'sequences': []
                        })
                        if not raw_sequences:
                            logger.debug(f"prebuild: no historical sequences for problem {pid}, using empty sequences")
                        if prompt_tokens is None:
                            logger.debug(f"prebuild: no prompt tokens for problem {pid}, using empty sequences")

                if problems_data:
                    try:
                        with_historical = sum(1 for problem_data in problems_data if problem_data['sequences'])
                        without_historical = len(problems_data) - with_historical
                        logger.info(f"🎯🎯🎯 PREBUILD EXECUTING: Total {len(problems_data)} problem_ids")
                        logger.info(f"🎯 - With historical data: {with_historical} problem_ids")
                        logger.info(f"🎯 - Without historical data (first training): {without_historical} problem_ids")
                        
                        # Print the actual problem_ids being prebuilt
                        prebuilt_pids = [problem_data['problem_id'] for problem_data in problems_data]
                        logger.info(f"🎯 Prebuilding problem_ids: {prebuilt_pids[:20]}{'...' if len(prebuilt_pids) > 20 else ''}")
                        
                        # Print per-problem sequence counts
                        pid_seq_counts = [(problem_data['problem_id'], len(problem_data['sequences'])) for problem_data in problems_data]
                        total_seqs = sum(cnt for _, cnt in pid_seq_counts)
                        avg_seqs = total_seqs / len(pid_seq_counts) if pid_seq_counts else 0
                        min_seqs = min((cnt for _, cnt in pid_seq_counts), default=0)
                        max_seqs = max((cnt for _, cnt in pid_seq_counts), default=0)
                        
                        logger.info(f"📊 Prebuild sequence stats: TOTAL={total_seqs}, AVG={avg_seqs:.1f}, MIN={min_seqs}, MAX={max_seqs}")
                        
                        # Sample detailed counts
                        sample_counts = pid_seq_counts[:3]
                        logger.info(f"📊 Sample prebuild counts: {sample_counts}")
                        
                        _log_performance_info("PREBUILD_EXECUTING", f"problems={len(problems_data)}")
                        suffix_cache.prebuild_problems_parallel(problems_data)
                        _log_performance_info("PREBUILD_COMPLETED", f"problems={len(problems_data)}")
                        logger.info(f"🎯✅ PREBUILD COMPLETED: Successfully prebuilt {len(problems_data)} problem_ids")
                    except Exception as e:
                        logger.error(f"prebuild execution failed: {e}")
                        _log_performance_info("PREBUILD_FAILED", f"error={str(e)[:50]}")
                else:
                    logger.warning("🎯❌ PREBUILD SKIPPED: no valid problems_data to process")
                    _log_performance_info("PREBUILD_SKIPPED", "no_valid_data")
            except Exception as e:
                logger.error(f"suffix_prebuild_loop error: {e}")
            finally:
                # Remove processed problem IDs from active set
                if 'problem_ids' in locals():
                    with self._prebuild_lock:
                        for pid in problem_ids:
                            self._active_prebuild_pids.discard(str(pid))
                
                self._suffix_prebuild_queue.task_done()
                logger.info(f"suffix_prebuild_loop task done, rank {rank}, local_rank {local_rank}")

    def _wait_for_prebuild_completion(self, problem_ids, timeout=30.0):
        """Wait for any ongoing prebuild tasks for the given problem_ids to complete.
        
        Args:
            problem_ids: List of problem IDs to wait for
            timeout: Maximum time to wait in seconds (default: 30s)
        """
        if not self.enable_suffix_prebuild or len(problem_ids) == 0:
            return
            
        problem_id_strs = set(str(pid) for pid in problem_ids)
        start_time = time.time()
        
        # Check initial state
        with self._prebuild_lock:
            initial_still_processing = problem_id_strs.intersection(self._active_prebuild_pids)
            if initial_still_processing:
                logger.info(f"🔄 Found {len(initial_still_processing)} problem_ids still being prebuilt, waiting...")
        
        while True:
            with self._prebuild_lock:
                # Check if any of our problem IDs are still being processed
                still_processing = problem_id_strs.intersection(self._active_prebuild_pids)
                if not still_processing:
                    break  # All our problem IDs are done
            
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"⏱️ Timeout waiting for prebuild completion of {len(still_processing)} problem_ids: {list(still_processing)[:10]}")
                break
                
            # Short sleep to avoid busy waiting
            time.sleep(0.1)
        
        elapsed = time.time() - start_time
        if elapsed > 0.1:  # Only log if we actually waited
            logger.info(f" Waited {elapsed:.2f}s for prebuild completion of {len(problem_ids)} problems")

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
        # ⏱️ Time profiling: Function start
        time_total_start = time.time()
        time_stage_start = time_total_start
        
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
            logger.warning(f"problem_id not found in batch, generated {batch_size} problem_ids")

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
            if SuffixDecodingCache is None:
                raise ImportError("SuffixDecodingCache not available. Please install arctic_inference package.")
            suffix_cache_max_depth = self.speculative_config.get("suffix_cache_max_depth", 64)
            suffix_cache_max_threads = self.speculative_config.get("suffix_cache_max_threads", None)

            # 🎯 SuffixCache线程数配置：硬编码为10线程
            if suffix_cache_max_threads is None:
                suffix_cache_max_threads = 10
                logger.info(f"🎯 SuffixCache use hardcoded threads: {suffix_cache_max_threads}")
            else:
                logger.info(f"🎯 SuffixCache use config specified threads: {suffix_cache_max_threads}")
            
            # 🚀 启用C++对象级锁定+GIL释放的线程安全模式
            # Reuse existing cache if available; otherwise create
            try:
                _existing = self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache
            except Exception:
                _existing = None
            if _existing is None:
                self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache = SuffixDecodingCache(
                    max_tree_depth=suffix_cache_max_depth, 
                    thread_safe=True, 
                    max_threads=suffix_cache_max_threads
                )
            
            if self.enable_suffix_prebuild:
                assert problem_ids is not None, "problem_ids must be provided when enable_suffix_prebuild is True"
                
                # 🔄 SYNC: Wait for any pending prebuild tasks for current problem_ids to complete
                _log_performance_info("SYNC_WAIT_START", f"pids={len(problem_ids)}")
                self._wait_for_prebuild_completion(problem_ids)
                _log_performance_info("SYNC_WAIT_END", f"pids={len(problem_ids)}")
                unique_problem_ids = list(set(problem_ids))
                logger.info(f"🔄✅ Prebuild wait completed for {len(problem_ids)} problem_ids, ({len(unique_problem_ids)} unique)")
            
                # Check if distribution_aware is enabled
                distribution_aware = self.speculative_config.get("distribution_aware", False)
                
                # Try reuse cached hard/medium sets from previous prebuild
                if distribution_aware and len(self._hard_pid_set) + len(self._medium_pid_set) > 0:        
                    hard_ids = [pid for pid in unique_problem_ids if str(pid) in self._hard_pid_set]
                    medium_ids = [pid for pid in unique_problem_ids if str(pid) in self._medium_pid_set]
                    easy_ids = [pid for pid in unique_problem_ids if str(pid) in self._easy_pid_set]
                    logger.info(f"generate_sequences (distribution_aware=True): using cached hard/medium/easy classification with length {len(hard_ids)} and {len(medium_ids)} and {len(easy_ids)}")
                else:
                    hard_ids = unique_problem_ids
                    medium_ids = []
                    easy_ids = []
                    if distribution_aware:
                        logger.info(f"generate_sequences (distribution_aware=True): no cached hard/medium sets, treating all {len(unique_problem_ids)} problems as medium difficulty")
                    else:
                        logger.info(f"generate_sequences (distribution_aware=False): treating all {len(unique_problem_ids)} problems as medium difficulty")
            else:
                unique_problem_ids = list(set(problem_ids))
                hard_ids = unique_problem_ids
                medium_ids = []
                easy_ids = []

        else:
            unique_problem_ids = list(set(problem_ids))
            hard_ids = unique_problem_ids
            medium_ids = []
            easy_ids = []

        # users can customize different sampling_params at different run
        local_rank = os.getenv("LOCAL_RANK", "0")
        rank = os.getenv("RANK", "0") 
        logger.debug(f"rank {rank}, local_rank {local_rank}, DEBUG: start generate sequences")
        with self.update_sampling_params(**kwargs):
            # Initialize context manager for problem_id to req_id mapping if problem_ids provided

            start_time = time.time()
            try:
                # Import ArcticInference plugin's ProblemIdContextManager
                from arctic_inference.vllm.model_runner import ProblemIdContextManager
                ProblemIdContextManager.clear_context()
                # Get current step from prompts.meta_info and set it in context manager
                current_step = prompts.meta_info.get("global_steps", None)
                assert current_step is not None, "global_steps not found in prompts.meta_info"
                ProblemIdContextManager.set_current_step(current_step)
                logger.debug(f"Set current_step={current_step} in ProblemIdContextManager")
                
                # Create empty req_id to problem_id mapping context
                ProblemIdContextManager.set_req_id_to_problem_id_mapping({})
                
                # Call generate with problem_ids parameter - LLM patches will handle the mapping
                # Also pass length_rank through the ProblemIdContextManager for this batch
                ProblemIdContextManager.set_hard_medium_ids(hard_ids, medium_ids,easy_ids)
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                    problem_ids=problem_ids,  # Pass problem_ids to generate method
                )
                end_time = time.time()
                logger.info(f"generate_sequences took {end_time - start_time:.2f} seconds")
            except (ImportError, TypeError):
                # Fallback if ArcticInference plugin is not available or LLM patches are disabled
                logger.warning("ArcticInference LLM plugin not available or disabled, problem_ids will be ignored")
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )
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

        # Update problem_id_to_files mapping if suffix prebuild is enabled
        if self.enable_suffix_prebuild:
            # Get current step from prompts.meta_info
            current_step = prompts.meta_info.get("global_steps", None)
            if current_step is None:
                logger.warning("global_steps not found in prompts.meta_info, skipping mapping update")
                return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            
            # Load mapping on first call when we have access to current_step
            if not self._mapping_loaded:
                start_time = time.time()
                self._load_problem_id_mapping_from_file(max_iteration=current_step)
                self._mapping_loaded = True
                end_time = time.time()
                logger.info(f"Loading problem_id mapping took {end_time - start_time:.2f} seconds for {current_step} steps")
            
            # Generate filename in format {global_step}.jsonl
            filename = f"{current_step}.jsonl"
            
            # Update the local mapping window
            self._update_problem_id_to_files_mapping(filename, current_step, problem_ids)
        
        # 批次级清理 - 清理所有suffix cache，避免内存泄漏
        time_cache_cleanup_start = time.time()
        if self.enable_suffix_prebuild:
            try:
                suffix_cache = self.inference_engine.llm_engine.model_executor.driver_worker.model_runner._suffix_cache
                if suffix_cache is not None:
                    # 记录清理前的缓存状态
                    problem_trees_before = len(suffix_cache._problem_tree) if hasattr(suffix_cache, '_problem_tree') else 0
                    prompt_trees_before = len(suffix_cache._prompt_trees) if hasattr(suffix_cache, '_prompt_trees') else 0
                    
                    logger.info(f"🧹 [Cache Cleanup] Starting cleanup: "
                               f"problem_trees={problem_trees_before}, "
                               f"prompt_trees={prompt_trees_before}")
                    
                    # 调用优化的清理方法（并行清理）
                    time_clear_start = time.time()
                    cleanup_result = suffix_cache.clear_all_cache()
                    time_clear = time.time() - time_clear_start
                    
                    if cleanup_result and isinstance(cleanup_result, dict) and cleanup_result.get("success"):
                        cleanup_method = cleanup_result.get("method", "unknown")
                        logger.info(f"⚡ [Cache Cleanup] Fast cleanup completed in {time_clear*1000:.2f}ms")
                        logger.info(f"⚡ [Cache Cleanup] Method: parallel cleanup using ThreadPoolExecutor")
                        logger.info(f"⚡ [Cache Cleanup] Synchronous parallel tree destruction completed")
                        logger.info(f"⚡ [Cache Cleanup] All trees cleared and memory freed")
                    else:
                        logger.warning(f"⚠️ [Cache Cleanup] Unexpected result: {cleanup_result}")
                        logger.info(f"⏱️ [Cache Cleanup] Cleanup time: {time_clear*1000:.2f}ms")
                    
                    # 验证清理后的状态（新的空缓存应该立即可用）
                    problem_trees_after = len(suffix_cache._problem_tree) if hasattr(suffix_cache, '_problem_tree') else 0
                    prompt_trees_after = len(suffix_cache._prompt_trees) if hasattr(suffix_cache, '_prompt_trees') else 0
                    
                    logger.info(f"✅ [Cache Cleanup] Cache state after cleanup: "
                               f"problem_trees: {problem_trees_before}→{problem_trees_after}, "
                               f"prompt_trees: {prompt_trees_before}→{prompt_trees_after}")
                    
            except Exception as e:
                logger.warning(f"❌ [Cache Cleanup] Failed to clear suffix cache: {e}")
                import traceback
                logger.warning(f"❌ [Cache Cleanup] Traceback: {traceback.format_exc()}")
        
        # ⏱️ Time profiling: Suffix cache cleanup
        time_cache_cleanup = time.time() - time_cache_cleanup_start
        logger.info(f"⏱️ [Profiling] Suffix cache cleanup took {time_cache_cleanup:.2f}s")
        
        # ⏱️ Time profiling: Total function time
        time_total = time.time() - time_total_start
        logger.info(f"⏱️ [Profiling] ===== TOTAL generate_sequences time: {time_total:.3f}s =====")

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
    