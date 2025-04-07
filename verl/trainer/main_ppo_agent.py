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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import ray
import hydra

# Local application imports
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.workers.reward_manager import NaiveRewardManager
from verl.trainer.ppo.ray_trainer_agent import RayPPOAgentTrainer

from rllm.environments.browsergym.browsergym import BatchBrowserGym, BrowserGym
from rllm.environments.frozenlake.frozenlake import BatchFrozenLakeEnv, FrozenLakeEnv
from rllm.environments.swe.swe import BatchSWEEnv, SWEEnv
from rllm.models.web_agent import WebAgent
from rllm.models.frozenlake_agent import FrozenLakeAgent
from rllm.models.swe_agent import SWEAgent

ENV_CLASS_MAPPING = {
    'browsergym': BrowserGym,
    'frozenlake': FrozenLakeEnv,
    'sweenv': SWEEnv,
}

AGENT_CLASS_MAPPING = {
    'webagent': WebAgent,
    'frozenlakeagent': FrozenLakeAgent,
    'sweagent': SWEAgent,
}

def setup_environment(config):
    if config.env.name == 'browsergym':
        if config.env.subtask == 'miniwob':
            import os
            import importlib
            import browsergym.miniwob
            importlib.reload(browsergym.miniwob)
            os.environ["MINIWOB_URL"] = config.env.miniwob_url
            return
    elif config.env.name == 'frozenlake':
        return
    elif config.env.name == "sweenv":
        return

    raise ValueError(f"Environment subtask not supported, env: {config.env.name}, subtask: {config.env.subtask == 'miniwob'}")


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo_agent(config)


def run_ppo_agent(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_local_path_from_hdfs
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


    reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    val_reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }
    
    # Below are agent specific initialization
    env_class = ENV_CLASS_MAPPING[config.env.name]
    agent_class = AGENT_CLASS_MAPPING[config.agent.name]
    setup_environment(config)    

    trainer = RayPPOAgentTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=RayWorkerGroup,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            env_class=env_class,
                            agent_class=agent_class)
    
    trainer.init_workers()
    trainer.fit_agent()
    


if __name__ == '__main__':
    main()
