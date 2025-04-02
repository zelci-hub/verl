import uuid
from pprint import pprint
from typing import Any, List, Type, Dict
import torch
from copy import deepcopy
import time
import threading
import queue

import numpy as np
from verl import DataProto

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    compute_timing_metrics, 
    compute_data_metrics,
    dataprotoitem_to_dataproto,
    compute_advantage,
    reduce_metrics,
)

class Timer:
    def __init__(self, name, timing_dict):
        self.name = name
        self.timing_dict = timing_dict
        self.start_time = None

    def __enter__(self):
        if self.name not in self.timing_dict:
            self.timing_dict[self.name] = 0.0
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        self.timing_dict[self.name] += elapsed

def update_metrics(metrics, new_metrics):
    for k, v in new_metrics.items():
        if k not in metrics:
            metrics[k] = []
        metrics[k].append(v)

class SortedQueue(queue.PriorityQueue):
    def __init__(self):
        super().__init__()

    def put(self, data):
        """Sort by attention mask sum before inserting into queue."""
        batch_size = data[-1].batch["attention_mask"].shape[0]
        priority = -data[-1].batch["attention_mask"].view(batch_size, -1).sum().item()
        super().put((priority, data))  # Priority queue sorts by first element

    def get(self):
        """Retrieve and return the highest-priority item."""
        return super().get()[1]

class RayPPOPipelineTrainer(RayPPOTrainer):
    
    def fit(self):
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
            val_metrics = self._validate()
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
                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                
                with Timer('step', timing_raw):

                    def create_replay_queue(generator, q):
                        with Timer('gen', timing_raw):
                            for gen_idx, item in enumerate(generator):
                                if item is None:
                                    return
                                q.put((batch_iter, gen_idx, item))     
                    # Get the generator function which will yield results as they complete
                    gen_seq_generator = self.rollout_wg.generate_sequences_async(prompts=batch)
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
                                print(f"Replay queue size: {replay_queue.qsize()}")
                                if replay_queue.qsize() >= ppo_mini_batch_size:
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
                            
                            
                            if self.config.actor_rollout_ref.rollout.enable_log_prob:
                                # Avoid recompute log_prob bugs. Log probs from vLLM. (Could be buggy)
                                mini_batch.meta_info['micro_batch_size'] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                                mini_batch.meta_info['max_token_len'] = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                                mini_batch.meta_info['use_dynamic_bsz'] = self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                                mini_batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
                            else:
                                # Recompute old_log_probs using Pytorch FSDP.
                                # This is also incorrect in the async version.
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
                                                  lam=self.config.algorithm.lam,)
                                                  #num_repeat=self.config.actor_rollout_ref.rollout.n)

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
                            val_metrics: dict = self._validate()
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
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with Timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
