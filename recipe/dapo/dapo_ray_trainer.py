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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        
        # rebuild problem_id mapping after checkpoint loading
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            enable_suffix_prebuild = self.config.trainer.get("enable_suffix_prebuild", False)
            if enable_suffix_prebuild:
                self._rebuild_problem_id_mapping_from_file(rollout_data_dir)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        # Initialize _next_batch for prebuild functionality
        self._next_batch = None
        dataloader_iter = iter(self.train_dataloader)
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, "green"):
                            save_token_ids = self.config.trainer.get("save_token_ids", False)
                            print("DEBUG:save_token_ids", save_token_ids)
                            if save_token_ids:
                                # Save token IDs (new behavior)
                                inputs = gen_batch_output.batch["prompts"].cpu().tolist()
                                outputs = gen_batch_output.batch["responses"].cpu().tolist()
                            else:
                                # Save decoded text (original behavior)
                                inputs = self.tokenizer.batch_decode(gen_batch_output.batch["prompts"], skip_special_tokens=True)
                                outputs = self.tokenizer.batch_decode(gen_batch_output.batch["responses"], skip_special_tokens=True)
                            
                            # Get problem_ids if available
                            problem_ids = gen_batch.non_tensor_batch.get("problem_id", [f"unknown_{i}" for i in range(len(inputs))])
                            
                            self._dump_generations_rollout(
                                inputs=inputs,
                                outputs=outputs,
                                problem_ids=problem_ids,
                                dump_path=rollout_data_dir,
                                save_token_ids=save_token_ids,
                            )

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # Log training generations with rewards if enabled
                    train_data_dir = self.config.trainer.get("train_data_dir", None)
                    if train_data_dir:
                        with marked_timer("dump_train_generations", timing_raw, "green"):
                            save_token_ids = self.config.trainer.get("save_token_ids", False)
                            if save_token_ids:
                                # Save token IDs
                                inputs = gen_batch_output.batch["prompts"].cpu().tolist()
                                outputs = gen_batch_output.batch["responses"].cpu().tolist()
                            else:
                                # Save decoded text
                                inputs = self.tokenizer.batch_decode(gen_batch_output.batch["prompts"], skip_special_tokens=True)
                                outputs = self.tokenizer.batch_decode(gen_batch_output.batch["responses"], skip_special_tokens=True)
                            
                            # Get scores
                            scores = new_batch.batch["token_level_scores"].sum(dim=-1).cpu().tolist()
                            
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=train_data_dir,
                                save_token_ids=save_token_ids,
                            )

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            # Try to get next batch for prebuild
                            try:
                                next_batch_dict = next(dataloader_iter)
                                self._next_batch = DataProto.from_single_dict(next_batch_dict)
                            except StopIteration:
                                self._next_batch = None
                            
                            self._prebuild_next_batch()
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    else:
                        # Actor update is skipped due to warmup; still trigger prebuild now
                        try:
                            next_batch_dict = next(dataloader_iter)
                            self._next_batch = DataProto.from_single_dict(next_batch_dict)
                        except StopIteration:
                            self._next_batch = None
                        self._prebuild_next_batch()

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                
                timing_raw = defaultdict(float)  # clear timing

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

    def _prebuild_next_batch(self) -> None:
        """Collect next-step problem IDs and prompt tokens and enqueue prebuild on rollout.

        If `self._next_batch` exists, extract `problem_id` and optional `raw_prompt_ids`.
        If `raw_prompt_ids` are absent, derive them by stripping left pads from `input_ids`.
        Then, hint rollout to prebuild via its non-blocking `enqueue_prebuild` RPC.
        """
        import logging
        
        if not getattr(self, "_next_batch", None):
            logging.warning("No next batch found, skipping prebuild")
            return
        try:
            next_ntb = self._next_batch.non_tensor_batch
            next_pids = next_ntb.get("problem_id", None)
            assert next_pids is not None, "problem_id is not found in next_batch for prebuild"
            next_pids = next_pids.tolist()
            next_raw = next_ntb.get("raw_prompt_ids", None)
            assert next_raw is not None, "raw_prompt_ids is not found in next_batch for prebuild"
            next_raw = next_raw.tolist()
            next_input_ids = self._next_batch.batch.get("input_ids", None)
            next_raw = []
            for i in range(next_input_ids.size(0)):
                seq = next_input_ids[i]
                next_raw.append(seq.tolist())

            # Directly hint rollout via non-blocking RPC, include current iteration
            try:
                current_iteration = int(self.global_steps)
                self.actor_rollout_wg.enqueue_prebuild(next_pids, next_raw, current_iteration)
            except Exception as e:
                print(f"WARN: enqueue_prebuild RPC failed: {e}")
        except Exception as e:
            print(f"WARN: _prebuild_next_batch failed: {e}")

    def _rebuild_problem_id_mapping_from_file(self, rollout_data_dir):
        """Rebuild problem_id mapping from existing file_to_problem_ids.jsonl file.
        
        Args:
            rollout_data_dir: Directory containing the file_to_problem_ids.jsonl file
        """
        file_to_problem_ids_file = os.path.join(rollout_data_dir, "file_to_problem_ids.jsonl")
        
        if not os.path.exists(file_to_problem_ids_file):
            print(f"No existing problem_id mapping file found at {file_to_problem_ids_file}")
            return
            
        try:
            with open(file_to_problem_ids_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        filename = entry.get("filename")
                        problem_ids = entry.get("problem_ids", [])
                        
                        # Add each problem_id to filename mapping
                        for pid in problem_ids:
                            if pid not in self._problem_id_to_files:
                                self._problem_id_to_files[pid] = []
                            if filename not in self._problem_id_to_files[pid]:
                                self._problem_id_to_files[pid].append(filename)
                                
            print(f"Rebuilt problem_id mapping from {file_to_problem_ids_file}")
            print(f"Loaded mappings for {len(self._problem_id_to_files)} problem_ids")
        except Exception as e:
            print(f"Warning: Failed to rebuild problem_id mapping: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, save_token_ids=False):
        """Dump rollout/validation samples as JSONL.
        
        Args:
            inputs: List of input prompts (text strings or token ID lists)
            outputs: List of output responses (text strings or token ID lists) 
            scores: List of reward scores
            reward_extra_infos_dict: Dictionary of additional info to save
            dump_path: Directory path to save the file
            save_token_ids: If True, inputs/outputs are token ID lists; if False, they are text strings
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.gen_steps}.jsonl")

        n = len(inputs)
        if save_token_ids:
            # Remove padding tokens from inputs and outputs
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            
            # Remove padding from inputs (left padding)
            cleaned_inputs = []
            for input_ids in inputs:
                if isinstance(input_ids, list):
                    # Find first non-pad token
                    first_non_pad_idx = 0
                    while first_non_pad_idx < len(input_ids) and input_ids[first_non_pad_idx] == pad_token_id:
                        first_non_pad_idx += 1
                    cleaned_inputs.append(input_ids[first_non_pad_idx:] if first_non_pad_idx < len(input_ids) else input_ids)
                else:
                    cleaned_inputs.append(input_ids)
            
            # Remove padding from outputs (right padding)
            cleaned_outputs = []
            for output_ids in outputs:
                if isinstance(output_ids, list):
                    # Find last non-pad token
                    last_non_pad_idx = len(output_ids) - 1
                    while last_non_pad_idx >= 0 and output_ids[last_non_pad_idx] == pad_token_id:
                        last_non_pad_idx -= 1
                    cleaned_outputs.append(output_ids[:last_non_pad_idx + 1] if last_non_pad_idx >= 0 else output_ids)
                else:
                    cleaned_outputs.append(output_ids)
            
            base_data = {
                "input_token_ids": cleaned_inputs,
                "output_token_ids": cleaned_outputs,
                "score": scores,
                "step": [self.gen_steps] * n,
            }
        else:
            base_data = {
                "input": inputs,
                "output": outputs,
                "score": scores,
                "step": [self.gen_steps] * n,
            }
        
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")
        
        # Update problem_id_to_files.jsonl mapping
        self._update_problem_id_mapping(dump_path, filename, reward_extra_infos_dict)

    def _update_problem_id_mapping(self, dump_path, filename, reward_extra_infos_dict):
        """Update file_to_problem_ids.jsonl
        
        Args:
            dump_path: Directory path where files are dumped
            filename: Full path to the dumped file
            reward_extra_infos_dict: Dictionary containing problem_id information
        """
        # Extract problem_ids from reward_extra_infos_dict
        problem_ids = reward_extra_infos_dict.get("problem_id", None)
        if problem_ids is None:
            return  # Skip mapping update if no problem_id available
        
        # Get just the filename (not full path) for storage
        base_filename = os.path.basename(filename)
        
        # Step 1: Append to file_to_problem_ids.jsonl
        file_to_problem_ids_file = os.path.join(dump_path, "file_to_problem_ids.jsonl")
        file_entry = {
            "filename": base_filename,
            "problem_ids": list(set(problem_ids)),  # Remove duplicates
            "step": self.gen_steps
        }
        
        try:
            with open(file_to_problem_ids_file, "a") as f:
                f.write(json.dumps(file_entry, ensure_ascii=False) + "\n")
        except IOError as e:
            print(f"Warning: Could not write to {file_to_problem_ids_file}: {e}")
            return

    def _dump_generations_rollout(self, inputs, outputs, problem_ids, dump_path, save_token_ids=False):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.gen_steps}.jsonl")
        
        n = len(inputs)
        if save_token_ids:
            # Remove padding tokens from inputs and outputs
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            
            # Remove padding from inputs (left padding)
            cleaned_inputs = []
            for input_ids in inputs:
                if isinstance(input_ids, list):
                    # Find first non-pad token
                    first_non_pad_idx = 0
                    while first_non_pad_idx < len(input_ids) and input_ids[first_non_pad_idx] == pad_token_id:
                        first_non_pad_idx += 1
                    cleaned_inputs.append(input_ids[first_non_pad_idx:] if first_non_pad_idx < len(input_ids) else input_ids)
                else:
                    cleaned_inputs.append(input_ids)
            
            # Remove padding from outputs (right padding)
            cleaned_outputs = []
            for output_ids in outputs:
                if isinstance(output_ids, list):
                    # Find last non-pad token
                    last_non_pad_idx = len(output_ids) - 1
                    while last_non_pad_idx >= 0 and output_ids[last_non_pad_idx] == pad_token_id:
                        last_non_pad_idx -= 1
                    cleaned_outputs.append(output_ids[:last_non_pad_idx + 1] if last_non_pad_idx >= 0 else output_ids)
                else:
                    cleaned_outputs.append(output_ids)
            
            base_data = {
                "input_token_ids": cleaned_inputs,
                "output_token_ids": cleaned_outputs,
                "problem_id": problem_ids,
                "step": [self.gen_steps] * n,
            }
        else:
            base_data = {
                "input": inputs,
                "output": outputs,
                "problem_id": problem_ids,
                "step": [self.gen_steps] * n,
            }
        
        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")
        
        # Update problem_id_to_files.jsonl mapping
        # Create a reward_extra_infos_dict-like structure for consistency
        reward_extra_infos_dict = {"problem_id": problem_ids}
        self._update_problem_id_mapping(dump_path, filename, reward_extra_infos_dict)
