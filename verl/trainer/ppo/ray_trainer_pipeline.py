import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, List, Type, Dict
import torch
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
# from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    _timer, 
    compute_timing_metrics, 
    compute_data_metrics,
    dataprotoitem_to_dataproto,
    compute_advantage,
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.torch_functional import masked_mean


class RayPPOPipelineTrainer(RayPPOTrainer):
    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
  
# use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        train_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        if self.config.trainer.rejection_sample:
            train_batch_size *= self.config.trainer.rejection_sample_multiplier
            train_batch_size = int(train_batch_size)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = self.config.rollout_ref.rollout.val_temperature
            test_output_gen_batch_padded = self.rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
		
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        def create_ray_class_with_cuda(role: str, cls, config: Any, **kwargs) -> RayClassWithInitArgs:
            # Pass CUDA_VISIBLE_DEVICES through kwargs
            # if gpu_ids:
            #     kwargs['cuda_visible_devices'] = ','.join(map(str, gpu_ids))
                
            worker_cls = RayClassWithInitArgs(
                cls=cls,
                config=config,
                **kwargs,
                role=role,
            )
            return worker_cls
			
		# Create actor and rollout
        if self.hybrid_engine:
            assert False, "This is only used with hybrid engine"
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            # gpu_ids = resource_pool.gpu_assignments[0] if isinstance(resource_pool, RayResourcePool) else None
            actor_rollout_cls = create_ray_class_with_cuda(
                'actor_rollout',
                self.role_worker_mapping[Role.ActorRollout],
                self.config.actor_rollout_ref,
            )
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            # Get actor resource pool
            actor_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
            # actor_gpu_ids = actor_resource_pool.gpu_assignments if isinstance(actor_resource_pool, RayResourcePool) else None
            actor_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Actor],
                config=self.config.actor_rollout_ref,
                role='actor',
            )
            self.resource_pool_to_cls[actor_resource_pool]['actor'] = actor_cls

            # Get rollout resource pool
            rollout_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
            # rollout_gpu_ids = rollout_resource_pool.gpu_assignments if isinstance(rollout_resource_pool, RayResourcePool) else None
            rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Rollout],
                config=self.config.actor_rollout_ref,
                role='rollout',
                # gpu_ids=rollout_gpu_ids,
            )
            self.resource_pool_to_cls[rollout_resource_pool]['rollout'] = rollout_cls

        # Create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            # gpu_ids = resource_pool.gpu_assignments if isinstance(resource_pool, RayResourcePool) else None
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=self.config.critic,
                role='critic',
                # gpu_ids=gpu_ids
            )
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # Create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            # gpu_ids = resource_pool.gpu_assignments if isinstance(resource_pool, RayResourcePool) else None
            ref_policy_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role='ref',
                # gpu_ids=gpu_ids,
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # Create reward model if needed
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            # gpu_ids = resource_pool.gpu_assignments if isinstance(resource_pool, RayResourcePool) else None
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
                role='rm',
                # gpu_ids=gpu_ids
            )
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        # Initialize models
        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # Store actor and rollout worker groups separately
        self.actor_wg = all_wg['actor']
        self.actor_wg.init_model()
        
        # Initialize models
        self.rollout_wg = all_wg['rollout']
        self.rollout_wg.init_model()

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_wg.load_checkpoint(actor_path)
        self.rollout_wg.load_checkpoint(actor_path)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        pending_updates = []
        mini_batch_replay_buffer = []
        for epoch in range(self.config.trainer.total_epochs):
            for mini_batch_idx, batch_dict in enumerate(self.train_dataloader):
                metrics = {}
                timing_raw = {}
                if mini_batch_idx > 0 and mini_batch_idx % int(self.config.data.train_batch_size/self.config.actor_rollout_ref.actor.ppo_mini_batch_size) == 0:
                    # flush all the updates
                    for actor_update_ref in pending_updates:
                        actor_output = actor_update_ref.get()
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    pending_updates = []
                    # submit all the replay buffers
                    for (buffer_batch, buffer_mini_batch_idx) in mini_batch_replay_buffer:
                        buffer_batch.meta_info["last_mini_batch"] = buffer_mini_batch_idx % int(self.config.data.train_batch_size/self.config.actor_rollout_ref.actor.ppo_mini_batch_size) == int(self.config.data.train_batch_size/self.config.actor_rollout_ref.actor.ppo_mini_batch_size) - 1
                        actor_update_ref = self.actor_wg.update_actor_mini_batch_step(buffer_batch)
                        actor_output = actor_update_ref.get()
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    mini_batch_replay_buffer = []
                    # TODO: update actor policy
                    updated_actor_module_fsdp = self.actor_wg.get_state_dict()[0]
                    print('haha ', len(self.actor_wg.get_state_dict()))
                    self.rollout_wg.update_rollout_actor_module(updated_actor_module_fsdp)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.rollout_wg.generate_sequences(gen_batch)

                    # This code matches a prompt ID with its N responses.
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                        
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_all'] = solve_all


                        if self.config.trainer.rejection_sample:
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]
                            batch = dataprotoitem_to_dataproto(batch)
                            # Round down to the nearest multiple of world size
                            num_trainer_replicas = self.actor_wg.world_size 
                            max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
                            if not max_batch_size:
                                # give up, you got everything either all wrong or right.
                                continue

                            size_mask = torch.zeros(batch.batch['input_ids'].shape[0], dtype=torch.bool)
                            size_mask[:max_batch_size] = True
                            batch = batch[size_mask]
                            batch = dataprotoitem_to_dataproto(batch)

                        # recompute old_log_probs
                        if self.config.actor_rollout_ref.rollout.vllm_log_prob:
                            # Avoid recompute log_prob bugs. Log probs from vLLM. (Could be buggy)
                            batch.meta_info['micro_batch_size'] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                            batch.meta_info['max_token_len'] = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                            batch.meta_info['use_dynamic_bsz'] = self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                            batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
                        else:
                            # Recompute old_log_probs using Pytorch FSDP.
                            with _timer('old_log_prob', timing_raw):
                                old_log_prob = self.actor_wg.compute_log_prob(batch)
                                batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with _timer('values', timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    mini_batch_replay_buffer.append((batch, mini_batch_idx))
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        if len(pending_updates) > 0:
                            assert len(pending_updates) == 1
                            finished = pending_updates[0].wait()
                            if finished:
                                # get the most recent ref
                                actor_output = pending_updates[0].get()
                                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                                metrics.update(actor_output_metrics)
                                # clear the pending_updates
                                pending_updates = pending_updates[1:]
                                # submit new batch
                                if len(mini_batch_replay_buffer) > 0:
                                    buffer_batch, buffer_mini_batch_idx = mini_batch_replay_buffer[0]
                                    mini_batch_replay_buffer = mini_batch_replay_buffer[1:]
                                    buffer_batch.meta_info["last_mini_batch"] = buffer_mini_batch_idx % int(self.config.data.train_batch_size/self.config.actor_rollout_ref.actor.ppo_mini_batch_size) == int(self.config.data.train_batch_size/self.config.actor_rollout_ref.actor.ppo_mini_batch_size) - 1
                                    actor_update_ref = self.actor_wg.update_actor_mini_batch_step(buffer_batch)
                                    pending_updates.append(actor_update_ref)
                        else:
                            # update actor
                            if len(mini_batch_replay_buffer) > 0:
                                buffer_batch, buffer_mini_batch_idx = mini_batch_replay_buffer[0]
                                mini_batch_replay_buffer = mini_batch_replay_buffer[1:]
                                buffer_batch.meta_info["last_mini_batch"] = buffer_mini_batch_idx % int(self.config.data.train_batch_size/self.config.actor_rollout_ref.actor.ppo_mini_batch_size) == int(self.config.data.train_batch_size/self.config.actor_rollout_ref.actor.ppo_mini_batch_size) - 1
                                actor_update_ref = self.actor_wg.update_actor_mini_batch_step(buffer_batch)
                                pending_updates.append(actor_update_ref)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
