import ray
import hydra

# Local application imports
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.reward_manager import NaiveRewardManager
from verl.trainer.ppo.ray_trainer_agent import RayPPOAgentTrainer
from verl.trainer.main_ppo_agent import ENV_CLASS_MAPPING, AGENT_CLASS_MAPPING, setup_environment
from verl.trainer.ppo.ray_trainer_agent_pipeline import RayPPOPipelineAgentTrainer

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo_agent_pipeline(config)


def run_ppo_agent_pipeline(config, compute_score=None):
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

    actor_pool_id = 'actor_pool'
    rollout_pool_id = 'rollout_pool'
    num_training_gpus = 2
    resource_pool_spec = {
        actor_pool_id: [num_training_gpus] * config.trainer.nnodes,
        rollout_pool_id: [config.trainer.n_gpus_per_node - num_training_gpus] * config.trainer.nnodes,
    }
    mapping = {
        Role.Actor: actor_pool_id,
        Role.Rollout: rollout_pool_id,
        Role.RefPolicy: actor_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


    reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    val_reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    role_worker_mapping = {
        Role.Actor: ray.remote(ActorRolloutRefWorker),
        Role.Rollout: ray.remote(ActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    
    # Below are agent specific initialization
    env_class = ENV_CLASS_MAPPING[config.env.name]
    agent_class = AGENT_CLASS_MAPPING[config.agent.name]
    setup_environment(config)    

    trainer = RayPPOPipelineAgentTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=RayWorkerGroup,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            env_class=env_class,
                            agent_class=agent_class,
                            agent_trajectory_episode_len=config.agent.trajectory_episode_len,
                            agent_safe_batch_size=config.agent.safe_batch_size)
    
    trainer.init_workers()
    trainer.fit_agent()
    


if __name__ == '__main__':
    main()
