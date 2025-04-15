"""
Generate trajectories
"""
import ray
import numpy as np
import hydra
import os
from tqdm import tqdm
import pandas as pd
import torch
import csv

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


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

from rllm.environments.browsergym.browsergym import BrowserGym
from rllm.models.web_agent import WebAgent
from rllm.models.agent_execution_engine import AgentExecutionEngine

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
            extra_infos = dataset["extra_info"].tolist()
            extra_infos = [x for x in extra_infos for _ in range(config.data.n_samples)]
            return [BrowserGym.from_extra_info(i) for i in extra_infos]

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
    envs = init_env(config)
    agent_class = AGENT_CLASS_MAPPING[config.agent.name]
    agents = [agent_class() for _ in range(len(envs))]

    agent_engine = AgentExecutionEngine(
        rollout_engine=rollout_engine,
        engine_name="verl",
        tokenizer=tokenizer,
        episode_len=config.agent.max_episodes,
        agents=agents,
        envs=envs
    )

    original_batch = DataProto.from_dict({"dummy_batch": torch.empty(len(envs), 1)})
    original_batch.meta_info = {
        "agent_rollout": True
    }

    evaluate_trajectories = agent_engine.interact_environment(original_batch=original_batch)

    evaluate_metrics = {
        "evaluate_rollout.mean": np.mean([
            d[0]["trajectory_reward"] if d else 0 
            for d in evaluate_trajectories
        ]),
        "evaluate_rollout.max": np.max([
            d[0]["trajectory_reward"] if d else 0 
            for d in evaluate_trajectories
        ]),
        "evaluate_rollout.min": np.min([
            d[0]["trajectory_reward"] if d else 0 
            for d in evaluate_trajectories
        ]),
    }

    print(evaluate_metrics)

    if config.data.output_metric_path:
        os.makedirs(os.path.dirname(config.data.output_metric_path), exist_ok=True)
        # Save to CSV file
        with open(config.data.output_metric_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in evaluate_metrics.items():
                writer.writerow([key, value])
        print("Metrics saved")

    return evaluate_trajectories


if __name__ == "__main__":
    main()
