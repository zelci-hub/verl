import concurrent.futures
from typing import Type, Dict, List
import numpy as np
import torch
from pprint import pprint
from omegaconf import OmegaConf, open_dict
import uuid
from functools import partial
from jinja2 import Template
import re

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    WorkerType,
    ResourcePoolManager,
    RayWorkerGroup,
    compute_timing_metrics, 
    compute_data_metrics,
    dataprotoitem_to_dataproto,
    compute_advantage,
    reduce_metrics,
    _timer,
)

from rllm.models.batch_agent import BatchAgent

class RayPPOAgentTrainer(RayPPOTrainer):

    def __init__(
            self,
            config,
            tokenizer,
            role_worker_mapping: dict[Role, WorkerType],
            resource_pool_manager: ResourcePoolManager,
            ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
            reward_fn=None,
            val_reward_fn=None,
            env_class=None,
            agent_class=None,
        ):
        super().__init__(config, tokenizer, role_worker_mapping,
                         resource_pool_manager, ray_worker_group_cls,
                         reward_fn, val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
    
    def init_workers(self):
        super().init_workers()

        assert not self.config.actor_rollout_ref.rollout.async_engine, "Must use synchronous engine for agent training"

        # Initialize additional agent class 
        # Number of agents is set to be 0 initially
        if self.hybrid_engine: 
            agent_rollout_wg = self.actor_rollout_wg
        else:
            agent_rollout_wg = self.rollout_wg
        
        self.agent = BatchAgent(
            rollout_engine=agent_rollout_wg,
            engine_name="verl",
            tokenizer=self.tokenizer,
            agent_class=self.agent_class,
            episode_len=self.config.agent.trajectory_episode_len,
        )


    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                ####################
                ####################
                # must pop those keys for generation so they no longer exist
                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                }
                # initialize environment
                env_ids = [
                    i["environment_id"] for i in batch.non_tensor_batch["extra_info"]
                ]
                env = self.env_class(batch_size=len(env_ids), env_id=env_ids)

                batch.non_tensor_batch["uid"] = np.array(env_ids, dtype=object)

                with _timer("step", timing_raw):
                    final_gen_batch_output = self.generate_agent_trajectory(env, timing_raw=timing_raw, meta_info=batch.meta_info)
                    env.close()

                    batch = batch.union(final_gen_batch_output)
                    ####################
                    ####################

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:  
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"] # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(
                                -1
                            )  # Sum rewards for each sequence

                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all

                        if self.config.trainer.rejection_sample:
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]
                            batch = dataprotoitem_to_dataproto(batch)
                            # Round down to the nearest multiple of world size
                            num_trainer_replicas = self.actor_rollout_wg.world_size
                            max_batch_size = (
                                batch.batch["input_ids"].shape[0]
                                // num_trainer_replicas
                            ) * num_trainer_replicas
                            if not max_batch_size:
                                # give up, you got everything either all wrong or right.
                                continue

                            size_mask = torch.zeros(
                                batch.batch["input_ids"].shape[0], dtype=torch.bool
                            )
                            size_mask[:max_batch_size] = True
                            batch = batch[size_mask]
                            batch = dataprotoitem_to_dataproto(batch)

                        # recompute old_log_probs
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch
                                )
                                batch = batch.union(ref_log_prob)

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

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=1,
                        )

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if (
                        self.config.trainer.save_freq > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        rewards_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "val_temperature": self.config.actor_rollout_ref.rollout.val_temperature,
                "agent_rollout": True
            }

            env_ids = [
                i["environment_id"] for i in test_batch.non_tensor_batch["extra_info"]
            ]

            env = self.env_class(batch_size=len(env_ids), env_id=env_ids)

            test_output_gen_batch = self.generate_agent_trajectory(
                env, meta_info=test_batch.meta_info
            )
            env.close()

            test_batch = test_batch.union(test_output_gen_batch)

            # use environment score to report validation reward
            reward_tensor = test_batch.batch["environment_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

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
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        return metric_dict

    def generate_agent_trajectory(self, env, timing_raw={}, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            env: The environment in which the agent interacts.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
        """
        # Reset the agent.
        self.agent.update_env(env)
        with _timer("collect_trajectory", timing_raw):
            # Interact_environment returns list of trajectories.
            trajectories = self.agent.interact_environment(
                timing_raw=timing_raw, meta_info=meta_info
            )

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_trajectories(
                trajectories
            )

        return final_gen_batch_output

    def _postprocess_model_chat_template(self, message_text):
        """
        Postprocesses the chat template output by removing any automatically added system message.

        Args:
            message_text (str): The formatted message text.

        Returns:
            str: The processed message text without the default system message.
        """
        if any(substring in self.config.actor_rollout_ref.model.path.lower() for substring in ('qwen2', 'qwen2.5')):
            # from https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json, a default system message is inserted. So we manually remove the first occurance of default system message.
            # This is currently assuming no tool call.
            target = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            if message_text.startswith(target):
                return message_text[len(target):]  # Remove only if it’s at the start
            return message_text

        print("Model not recognized for postprocessing, entire text returned")
        return message_text

            
        
    def _transform_single_trajectory(self, traj_id, traj):
        from rllm.environments import compute_training_score, compute_environment_score
        """
        Transforms a single trajectory into tokenized form.

        Args:
            traj_idx: The index of the agent from which the trajectory originates.
            traj (list[dict]): The trajectory data, each step containing keys:
                "observation", "next_observation", "reward", "done", "action", "response", "augmented_reward".

        Returns:
            Tuple of torch tensors containing:
            - Initial prompt tokens
            - Response tokens
            - Full trajectory tokens (concatenation of prompt and response)
            - Mask for responses
            - Score values
            - Score positions (positions of last valid output token for each step)
            - Environment score
            - Environment score position (which is same as last reward position)
        """

        assert (
            traj and "observation" in traj[0]
        ), f"Trajectory is in wrong format. traj: {traj}, traj[0]: {traj[0]}"

        # Build initial prompts
        initial_message = {
            "role": "user",
            "content": self.agent.agents[traj_id].convert_observation_to_string(traj[0]["observation"], with_system_prompt=True),
        }
        inital_message_text = self.tokenizer.apply_chat_template(
            [initial_message], tokenize=False, add_generation_prompt=False
        )   
        inital_message_text = self._postprocess_model_chat_template(inital_message_text)
        
        prompts_tokens = self.tokenizer.encode(
            inital_message_text, add_special_tokens=False
        )

        # Build the trajectory messages
        traj_message = []
        
        for step in traj:
            next_obs = step["next_observation"]
            response = step["response"]
            traj_message.append({"role": "assistant", "content": response})
            observation_content = self.agent.agents[traj_id].convert_observation_to_string(next_obs, with_system_prompt=False)
            traj_message.append(
                {"role": "user", "content": observation_content}
            )

        # Convert the traj_message to inputs and mask tensors
        all_response_tokens = []
        all_response_masks = []
        for msg in traj_message:
            # Get template for single message
            msg_text = self.tokenizer.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=False
            )
            msg_text = self._postprocess_model_chat_template(msg_text)

            msg_tokens = self.tokenizer.encode(msg_text, add_special_tokens=False)

            mask_value = 1 if msg["role"] == "assistant" else 0
            msg_mask = [mask_value] * len(msg_tokens)

            all_response_tokens.extend(msg_tokens)
            all_response_masks.extend(msg_mask)


        traj_score = compute_training_score(traj)
        env_score = compute_environment_score(traj)

        return (
            torch.tensor(prompts_tokens, dtype=torch.long),
            torch.tensor(all_response_tokens, dtype=torch.long),
            torch.tensor(all_response_masks, dtype=torch.long),
            traj_score,
            env_score,
        )

    def _transform_agent_trajectories_helper(self, trajectories: List[List[Dict]]):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        # Start threading for parallel execution
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
            transform_func = partial(self._transform_single_trajectory)
            results = list(
                executor.map(transform_func, list(range(len(trajectories))), trajectories)
            )

        (
            all_initial_tokens_list,
            all_response_tokens_list,
            all_masks_list,
            traj_scores,
            environment_scores,
        ) = zip(*results)

        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list], 
            batch_first=True,  
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])                

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        traj_mask = torch.nn.utils.rnn.pad_sequence(
            all_masks_list, batch_first=True, padding_value=0
        )

        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)

        # Compute position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        environment_score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score
                environment_score_batch[i, last_valid_idx] = environment_scores[i]

        print(f"Shapes after convertion: complete trajectory: {trajectory_batch.size()}, responses: {response_batch.size()}, prompts: {prompts_batch.size()}, traj_mask: {traj_mask.size()}")

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "traj_mask": traj_mask,
            "environment_scores": environment_score_batch,
        }

        return DataProto.from_dict(tensors=tensor_batch)
    

    def _transform_agent_trajectories(
        self, trajectories: List[List[Dict]]
    ) -> DataProto:
        """
        Transforms a batch of agent trajectories into a structured DataProto format.

        Args:
            trajectories (list of list of dict): A batch of agent trajectories.

        Returns:
            DataProto: The structured dataset containing tokenized trajectories, masks, and rewards.
        """
        return self._transform_agent_trajectories_helper(trajectories=trajectories)
