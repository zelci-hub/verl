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
Generate responses given a dataset of prompts
"""
import csv
import ray
import json
import numpy as np
import hydra
import os
import torch
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score import _default_compute_score

import pandas as pd

from transformers import AutoTokenizer
import wandb

from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import (
    DataProtoItem,
    pad_dataproto_to_divisor,
    unpad_dataproto,
)
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.trainer.ppo.ray_trainer import dataprotoitem_to_dataproto
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.reward_manager import NaiveRewardManager

# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from rllm.rewards.rl_reward import rllm_reward_fn
        reward_fn = lambda s, gt: rllm_reward_fn(data_source, s, gt)
        return reward_fn

# This function is used to compute the score for each response, keeping the index
def compute_scores(idx, data_source, solution_lst, ground_truth, extra_info=None):
    rewards_computed = []
    for solution_str in solution_lst:
        rewards_computed.append(_default_compute_score(data_source, solution_str, ground_truth, extra_info))
    return idx, rewards_computed

def save_to_parquet(data, output_path, save_to_json=False):
    if not os.path.exists(os.path.dirname(output_path)):
        makedirs(os.path.dirname(output_path))
    data.to_parquet(output_path)
    if save_to_json:
        data.to_json(output_path.replace('.parquet', '.json'), orient='records')
    print(f"Saved to {output_path}")

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    wandb.init(project='verl')
    run_generation(config)


def run_generation(config):
    assert config.rollout.compute_reward is False, 'compute_reward must be False for generation'
    assert config.rollout.enable_log_prob is False, 'enable_log_prob must be False for generation'
    assert config.rollout.async_engine is False, 'async_engine must be False for generation'

    dataset_name = os.path.basename(config.data.path)

    local_path = copy_to_local(config.model.path)

    tokenizer = hf_tokenizer(local_path)

    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)

    n_val = config.rollout.n_val
    
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists, skipping generation")
        # Get dataset name from the path
        output_file = config.data.output_path
        
        try:
            existing_data = pd.read_parquet(output_file)
        except Exception as e:
            existing_data = pd.read_json(output_file.replace('.parquet', '.json'))
   
        existing_data['scores'] = None

        # preserve only the 4 args we need for compute_scores
        existing_data_args = existing_data[['data_source', 'responses', 'reward_model', 'extra_info']]
        existing_data_args['reward_model'] = existing_data_args['reward_model'].apply(lambda x: x['ground_truth'])
        
        # Convert the DataFrame to a list of lists for easier processing
        responses_lst = existing_data_args.values.tolist()

        scores = []

        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(compute_scores, i, *responses_lst[i]) for i in range(len(responses_lst))]
            for future in as_completed(futures):
                try:
                    idx, score_lst = future.result()
                    # Add the score to the existing data
                    existing_data.at[idx, 'scores'] = score_lst
                    scores.append((idx, score_lst))
                except Exception as e:
                    print(f"Error processing item: {e}")
        
        scores = list(map(lambda x: x[1], sorted(scores, key=lambda x: x[0])))

        # save the existing_data back to the output file
        save_to_parquet(existing_data, config.data.output_path, save_to_json=True)

    else:
        if config.rollout.temperature == 0.:
            assert config.rollout.n_val == 1, 'When temperature=0, n_val must be 1.'

        # Load Dataset
        try:
            dataset = pd.read_parquet(config.data.path)
        except Exception as e:
            config.data.path = config.data.path.replace('.parquet', '.json')
            print(f"Error loading dataset: {e}")
            dataset = pd.read_json(config.data.path)
            

        val_dataset = RLHFDataset(parquet_files=config.data.path,
            tokenizer=tokenizer,
            prompt_key=config.data.prompt_key,
            max_prompt_length=config.rollout.prompt_length,
            return_raw_chat=config.data.get('return_raw_chat', False),
            truncation='error')
        val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,
            collate_fn=collate_fn)
        
        # Load Rollout Workers
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='actor_rollout')
        rollout_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        rollout_wg.init_model()
    
        for test_data in val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            n_val = config.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val, interleave=True)
            
            # Store original inputs
            test_batch.meta_info = {
                'eos_token_id': tokenizer.eos_token_id,
                'pad_token_id': tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            test_batch_padded, pad_size = pad_dataproto_to_divisor(test_batch, rollout_wg.world_size)
            test_output_gen_batch_padded = rollout_wg.generate_sequences(test_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)    
        
        output_ids = test_output_gen_batch_padded.batch['responses']
        
        test_batch = dataprotoitem_to_dataproto(test_output_gen_batch) #test_batch.union(test_output_gen_batch)
        
        output_lst =  tokenizer.batch_decode(test_batch.batch['input_ids'][:, -config.rollout.response_length:], skip_special_tokens=False)
        output_lst = np.array(output_lst).reshape(len(output_lst)//n_val, n_val).tolist()
        dataset['responses'] = output_lst
        
        save_to_parquet(dataset, config.data.output_path)

        # evaluate using reward_function
        val_reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=1)
        # Runs reward calculation on parallel threads.
        reward_tensor = val_reward_fn(test_batch)
        
        scores = reward_tensor.sum(-1).cpu().tolist()
        scores = np.array(scores).reshape(len(scores)//n_val, n_val).tolist()
        dataset['scores'] = scores

        save_to_parquet(dataset, config.data.output_path, save_to_json=True)
            
    
    # Max across first dim average correct
    scores = np.array(scores)
    pass_at_n = np.max(scores, axis=1).mean()
    pass_at_1 = np.mean(scores)

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'pass.csv')
    results_path = os.path.join(output_dir, 'results.json')
    
    # Prepare the row data
    # Extract the dataset name from the path
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_val}': pass_at_n
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    with open(results_path, 'w') as f:
        json.dump(scores.tolist(), f)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

if __name__ == '__main__':
    main()
