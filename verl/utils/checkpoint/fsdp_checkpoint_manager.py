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

import glob
import ray
import os

import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig, ShardedOptimStateDictConfig
from safetensors.torch import load_file

from verl.utils.fs import copy_to_local, is_non_local

from transformers import PreTrainedTokenizer

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save 
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(self, model: FSDP, optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        super().__init__(model, optimizer, lr_scheduler, tokenizer)

    def load_checkpoint(self, path=None, del_local_after_load=False, *args, **kwargs):
        if path is None:
            return
        # every rank download its own checkpoint
        remote_model_path = os.path.join(path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
        checkpoint_model_path = os.path.join(path, f'checkpoint')
        remote_optim_path = os.path.join(path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
        remote_extra_state_path = os.path.join(path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')
        print(
            f'[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}'
        )
        local_model_path = copy_to_local(remote_model_path)
        checkpoint_model_path = copy_to_local(checkpoint_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(
                    f'[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored'
                )

        # Check which type of model checkpoint exists and load accordingly
        if False and os.path.exists(checkpoint_model_path):
            # Load the full model state
            print(f'[rank-{self.rank}]: Loading full model state from {checkpoint_model_path}')
            model_state_dict = {}
            if hasattr(self.model, '_fsdp_wrapped_module'):
                module = self.model._fsdp_wrapped_module
                # Glob all the safe tensor files
                safe_files = glob.glob(os.path.join(checkpoint_model_path, '*.safetensors'))
                for safe_file in safe_files:
                    state_dict = load_file(safe_file)
                    model_state_dict.update(state_dict)
            else:
                raise NotImplementedError("Only support FSDP wrapped model for now")
            
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, state_dict_config=full_state_dict_config):
                self.model.load_state_dict(model_state_dict)
            
            optimizer_state_dict = None
            if self.optimizer is not None:
                optimizer_state_dict = torch.load(local_optim_path)
                optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
                with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, optim_state_dict_config=optim_cfg):
                    self.optimizer.load_state_dict(optimizer_state_dict)


        else:
            # Load the sharded model state
            print(f'[rank-{self.rank}]: Loading sharded model state from {local_model_path}')
            model_state_dict = torch.load(local_model_path)
            state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_config=state_dict_cfg, optim_state_dict_config=optim_cfg):
                self.model.load_state_dict(model_state_dict)
                if self.optimizer is not None:
                    optimizer_state_dict = torch.load(local_optim_path)
                    self.optimizer.load_state_dict(optimizer_state_dict)
        
        extra_state_dict = None
        if self.lr_scheduler is not None:
            extra_state_dict = torch.load(local_extra_state_path)

        lr_scheduler_state_dict = None
        if self.lr_scheduler is not None:
            lr_scheduler_state_dict = extra_state_dict['lr_scheduler']
        # recover random state
        if extra_state_dict is not None and 'rng' in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict['rng'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

        del model_state_dict
        del optimizer_state_dict
        del extra_state_dict
        torch.cuda.empty_cache()

        # wait for everyone to load
        torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, global_step: int, remove_previous_ckpt=False, *args, **kwargs):
        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        # TODO: shall we remove previous ckpt every save?
        if remove_previous_ckpt:
            self.remove_previous_save_local_path()
        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                if self.optimizer is not None:
                    optimizer_state_dict = self.optimizer.state_dict()
                else:
                    optimizer_state_dict = None
                if self.lr_scheduler is not None:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict()
                else:
                    lr_scheduler_state_dict = None

                extra_state_dict = {
                    'lr_scheduler': lr_scheduler_state_dict,
                    'rng': self.get_rng_state(),
                }
                model_path = os.path.join(local_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
                optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
                extra_path = os.path.join(local_path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')

                print(f'[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}')
                print(f'[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(model_path)}')
                print(f'[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}')
                torch.save(model_state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.rank == 0:
            hf_local_path = os.path.join(local_path, 'huggingface')
            os.makedirs(hf_local_path, exist_ok=True)
            self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
            self.tokenizer.save_pretrained(hf_local_path)

        torch.distributed.barrier()

        self.previous_save_local_path = local_path
