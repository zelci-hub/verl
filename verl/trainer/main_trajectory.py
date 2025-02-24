"""
Generate trajectories
"""
import ray
import numpy as np
import hydra
import os
from tqdm import tqdm
import pandas as pd

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import gymnasium as gym

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)

from rllm.environments.browsergym import BatchBrowserGym
from rllm.models.web_agent import WebAgent
from rllm.models.batch_agent import BatchAgent

AGENT_CLASS_MAPPING = {
    'webagent': WebAgent,
}

def init_rollout_engine(config):
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout"
    )
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
    )
    wg = RayWorkerGroup(
        resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init
    )
    wg.init_model()
    return wg


def init_env(config):
    # Init env
    if config.env.name == "browsergym":
        if config.env.subtask == "miniwob":
            import os
            import importlib
            import browsergym.miniwob

            importlib.reload(browsergym.miniwob)
            os.environ["MINIWOB_URL"] = config.env.miniwob_url

            dataset = pd.read_parquet(config.data.path)
            selected_envs = [
                d["environment_id"] for d in dataset["extra_info"].tolist()
            ]
            selected_envs = selected_envs[:2]

            return BatchBrowserGym(
                env_id=selected_envs,
                batch_size=len(selected_envs),
            )

    raise ValueError(f"Environment {config.env.name} not supported")


@hydra.main(config_path="config", config_name="trajectory", version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rollout_engine = init_rollout_engine(config)
    env = init_env(config)
    agent_class = AGENT_CLASS_MAPPING[config.agent.name]

    agent = BatchAgent(
        rollout_engine=rollout_engine,
        engine_name="verl",
        tokenizer=tokenizer,
        agent_class=agent_class,
        n_parallel_agents=env.batch_size,
        env=env,
        safe_batch_size=config.agent.safe_batch_size,
        episode_len=config.agent.trajectory_episode_len,
    )
    evaluate_trajectories = agent.interact_environment()
    env.close()
    return evaluate_trajectories


if __name__ == "__main__":
    main()
