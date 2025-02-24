import uuid
from pprint import pprint
import torch
from copy import deepcopy
import time
import threading
import queue

import numpy as np
from verl import DataProto

from verl.trainer.ppo.ray_trainer import (
    compute_timing_metrics, 
    compute_data_metrics,
    compute_advantage,
    reduce_metrics,
)

from verl.trainer.ppo.ray_trainer_agent import (
    RayPPOAgentTrainer, 
)

from verl.trainer.ppo.ray_trainer_pipeline import (
    Timer,
    update_metrics,
    SortedQueue,
)

from rllm.models.batch_agent import BatchAgent

class RayPPOPipelineAgentTrainer(RayPPOAgentTrainer):

    def init_workers(self):
        super(RayPPOAgentTrainer, self).init_workers() # use init_workers from RayTrainer
        # Initialize additional agent class 
        assert not self.hybrid_engine, "Pipeline Agent only support non-hybrid engine"
        if self.hybrid_engine: 
            agent_rollout_wg = self.actor_rollout_wg
        else:
            agent_rollout_wg = self.rollout_wg

        self.agent = BatchAgent(
            rollout_engine=self.actor_rollout_wg,
            engine_name="verl",
            tokenizer=self.tokenizer,
            agent_class=self.agent_class,
            episode_len=self.agent_trajectory_episode_len,
            safe_batch_size=self.agent_safe_batch_size,
        )

    def validate_agent(self):
        rewards_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(['input_ids', 'attention_mask', 'position_ids']) # these are not needed for environment based interaction
            test_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            env_ids = [x['environment_id'] for x in test_batch.non_tensor_batch['extra_info']]
            env = self.env_class(batch_size = len(env_ids), env_id = env_ids)
            # Reset the agent.
            self.agent.reset_and_update_number_of_agents(len(env_ids))

            test_output_gen_batch = self.agent.generate_agent_trajectory(env, original_batch=test_batch)

            if test_batch.meta_info["recompute_log_prob"]:
                with torch.no_grad():
                    output = self.actor_rollout_wg.compute_log_prob(test_output_gen_batch)
                    test_output_gen_batch = test_output_gen_batch.union(output)

            test_batch = test_batch.union(test_output_gen_batch)

            # already evaluated during transformation
            reward_tensor = test_batch.batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
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

    def fit_agent(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        assert not self.hybrid_engine, "PPO pipeline trainer does not support hybrid engine, assumes Rollout and Actor are in the same worker group"

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
            val_metrics = self.validate_agent()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        replay_queue = SortedQueue() #queue.Queue()
        total_mini_batch_iters = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_iter, batch_dict in enumerate(self.train_dataloader):
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
    
                env_ids = [x['environment_id'] for x in batch.non_tensor_batch['extra_info']]
                env = self.env_class(batch_size = len(env_ids), env_id = env_ids)
                # Reset the agent.
                self.agent.reset_and_update_number_of_agents(len(env_ids))
                batch.non_tensor_batch['uid'] = np.array(env_ids, dtype=object)
                
                with Timer('step', timing_raw):

                    def create_replay_queue(generator, q):
                        with Timer('gen', timing_raw):
                            for gen_idx, item in enumerate(generator):
                                if item is None:
                                    return
                                q.put((batch_iter, gen_idx, item))     
                    # Get the generator function which will yield results as they complete
                    gen_seq_generator = self.agent.generate_trajectories_async(env, timing_raw=timing_raw, original_batch=batch)
                    thread = threading.Thread(target=create_replay_queue, args=(gen_seq_generator, replay_queue))
                    thread.start()
                    
                    ppo_train_batch_size = self.config.data.train_batch_size
                    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    assert ppo_train_batch_size % ppo_mini_batch_size == 0, "PPO mini batch size must be a divisor of the total training batch size"
                    ppo_step_minibatch_iter = ppo_train_batch_size // ppo_mini_batch_size
                    num_loops = ppo_step_minibatch_iter +1 if batch_iter > 0 else  ppo_step_minibatch_iter 
                    # Initialize Empty data proto
                    training_batch = []
                    for mini_batch_iter in range(num_loops):
                        
                        if mini_batch_iter == num_loops - 1:
                            while True:
                                if replay_queue.qsize() == ppo_mini_batch_size:
                                    break
                                time.sleep(1)
                            break
                        mini_batch_metrics = {}
                        start_time = time.perf_counter()         
                        with Timer('pipeline_gen', timing_raw):
                            outputs = []
                            for _ in range(ppo_mini_batch_size):
                                _, _, output = replay_queue.get()
                                outputs.append(output)
                            mini_batch = DataProto.concat(outputs)
                        end_time = time.perf_counter()
                        print(f"Generate mini batch took {end_time - start_time:.2f} seconds")
                        
                        if total_mini_batch_iters % ppo_step_minibatch_iter == ppo_step_minibatch_iter - 1:
                            mini_batch.meta_info['last_mini_batch'] = True

                        with Timer('adv', timing_raw):
                            reward_tensor = self.reward_fn(mini_batch)
                            print('Reward tensor:', reward_tensor.sum(-1))
                            mini_batch.batch['token_level_scores'] = reward_tensor
                        
                                                # Rejection sampling based on rewards
                            # Group rewards by uid
                            uids = mini_batch.non_tensor_batch['uid']
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
                            mini_batch_metrics['batch/solve_none'] = solve_none
                            mini_batch_metrics['batch/solve_all'] = solve_all
                            
                            
                            if self.config.actor_rollout_ref.rollout.vllm_log_prob:
                                # Avoid recompute log_prob bugs. Log probs from vLLM. (Could be buggy)
                                mini_batch.meta_info['micro_batch_size'] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                                mini_batch.meta_info['max_token_len'] = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                                mini_batch.meta_info['use_dynamic_bsz'] = self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                                mini_batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
                            else:
                                # Recompute old_log_probs using Pytorch FSDP.
                                with Timer('old_log_prob', timing_raw):
                                    old_log_prob = self.actor_wg.compute_log_prob(mini_batch)
                                    mini_batch = mini_batch.union(old_log_prob)

                            if self.use_reference_policy:
                                # compute reference log_prob
                                with Timer('ref', timing_raw):
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(mini_batch)
                                    mini_batch = mini_batch.union(ref_log_prob)
                            
                            mini_batch.batch['token_level_rewards'] = mini_batch.batch['token_level_scores']
                                                    # compute advantages, executed on the driver process
                            mini_batch = compute_advantage(mini_batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                        self._balance_batch(mini_batch, metrics=mini_batch_metrics)
                        # compute global_valid tokens
                        mini_batch.meta_info['global_token_num'] = torch.sum(mini_batch.batch['attention_mask'], dim=-1).tolist()
                        # update actor
                        start_time = time.perf_counter()
                        with Timer('update_actor', timing_raw):
                            actor_output = self.actor_wg.update_actor_mini_batch(mini_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        end_time = time.perf_counter()
                        print(f"Actor update took {end_time - start_time:.2f} seconds")
                        mini_batch_metrics.update(actor_output_metrics)
                        training_batch.append(mini_batch)
                        update_metrics(metrics, mini_batch_metrics)
                        total_mini_batch_iters += 1

                    # last_iter_mini_batch_iter = (mini_batch_iter + last_iter_mini_batch_iter - 1) % ppo_step_minibatch_iter
                    with Timer('rollout_model_update', timing_raw):
                        updated_actor_module_fsdp_ref = self.actor_wg.get_state_dict()
                        if isinstance(updated_actor_module_fsdp_ref, list):
                            updated_actor_module_fsdp_ref = updated_actor_module_fsdp_ref[0]
                        self.rollout_wg.update_rollout_actor_module(updated_actor_module_fsdp_ref)
                    training_batch = DataProto.concat(training_batch)
                    
                    # Validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with Timer('testing', timing_raw):
                            val_metrics: dict = self.validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with Timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()    
                
                metrics.update(compute_data_metrics(batch=training_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=training_batch, timing_raw=timing_raw))
                
                
                for k, v in metrics.items():
                    if isinstance(v, (list, np.ndarray)):
                        if 'batch/' in k:
                            metrics[k] = np.sum(v)
                        else:
                            metrics[k] = np.mean(v)
                
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1


                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self.validate_agent()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with Timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return