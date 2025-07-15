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
FSDP Exploration Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import time
import uuid
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, field
from datasets import Dataset
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.explore import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
import copy

from transformers import (
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean
torch.set_printoptions(threshold=10_000)


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_format_score = batch.batch['token_level_format_scores'].sum(-1)
    sequence_correctness_score = batch.batch['token_level_correctness_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    if use_critic:
        values = batch.batch['values']
        valid_values = values[:, 0]

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # format score
        'critic/format_score/mean':
            torch.mean(sequence_format_score).detach().item(),
        'critic/format_score/max':
            torch.max(sequence_format_score).detach().item(),
        'critic/format_score/min':
            torch.min(sequence_format_score).detach().item(),
        # correctness_score
        'critic/correctness_score/mean':
            torch.mean(sequence_correctness_score).detach().item(),
        'critic/correctness_score/max':
            torch.max(sequence_correctness_score).detach().item(),
        'critic/correctness_score/min':
            torch.min(sequence_correctness_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }

def push_buffer_to_hub(replay_buffer, replay_buffer_records, replay_buffer_g_x, repo_name):

    buffer_to_hub = {
        'index': [],
        'prompt': [],
        'correct_ratio': [],
        'records': [],
        'g(x)': [],
    }

    for i in replay_buffer.keys():
        buffer_to_hub['index'].append(replay_buffer[i].non_tensor_batch['extra_info']['index'])
        buffer_to_hub['prompt'].append(replay_buffer[i].non_tensor_batch['extra_info']['question'])

        records = np.array(replay_buffer_records[i]).astype(int)
        buffer_to_hub['correct_ratio'].append(records.mean())
        buffer_to_hub['records'].append(records)

        buffer_to_hub['g(x)'].append(replay_buffer_g_x[i])

    buffer_to_hub = Dataset.from_dict(buffer_to_hub)
    buffer_to_hub.push_to_hub(f"GitBag/{repo_name}")


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayExploreTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         eta=self.config.data.eta,
                                         filter_correct=self.config.data.filter_correct,
                                         filter_incorrect=self.config.data.filter_incorrect,
                                         num_gen_to_use=self.config.data.num_gen_to_use,
                                         normalize_rw=self.config.reward_model.normalize,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         add_prompt_template=self.config.data.add_prompt_template,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=int(self.config.data.train_batch_size * self.config.data.filter_time),
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       eta=-1,
                                       filter_correct=False,
                                       filter_incorrect=False,
                                       num_gen_to_use=0,
                                       normalize_rw=self.config.reward_model.normalize,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       add_prompt_template=self.config.data.add_prompt_template,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = int(len(self.train_dataloader) * self.config.trainer.total_epochs * self.config.data.filter_time)

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        format_score_tensor_lst, correctness_score_tensor_lst = [], []
        data_source_lst = []
        recorded_sequence_reward_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            format_score_tensor, correctness_score_tensor, recorded_sequence_reward = self.val_reward_fn(test_batch)

            format_score_tensor_lst.append(format_score_tensor)
            correctness_score_tensor_lst.append(correctness_score_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * format_score_tensor.shape[0]))
            recorded_sequence_reward_lst.append(recorded_sequence_reward)

        format_score_tensor = torch.cat(format_score_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        correctness_score_tensor = torch.cat(correctness_score_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        metric_dict = {}

        # log wandb table
        recorded_sequence_reward_lst = np.concatenate(recorded_sequence_reward_lst, axis=0)
        
        if 'wandb' in self.config.trainer.logger and self.config.trainer.log_table_val != 0:
            import wandb
            if self.config.trainer.log_table_val < format_score_tensor.shape[0] and self.config.trainer.log_table_val > 0:
                log_data_sources = data_sources[:self.config.trainer.log_table_val]
                log_sequences_str = recorded_sequence_reward_lst[:self.config.trainer.log_table_val, 0]
                log_ground_truth = recorded_sequence_reward_lst[:self.config.trainer.log_table_val, 1]
                log_format_score = recorded_sequence_reward_lst[:self.config.trainer.log_table_val, 2]
                log_correctness_score = recorded_sequence_reward_lst[:self.config.trainer.log_table_val, 3]
            else:
                log_data_sources = data_sources
                log_sequences_str = recorded_sequence_reward_lst[:, 0]
                log_ground_truth = recorded_sequence_reward_lst[:, 1]
                log_format_score = recorded_sequence_reward_lst[:, 2]
                log_correctness_score = recorded_sequence_reward_lst[:, 3]

            metric_dict[f'val/generations_step_{self.global_steps}'] = wandb.Table(columns=["Data Source", "Sequence", "Ground Truth", "Format Score", "Correctness Score"], \
                                                                                   data=np.stack([log_data_sources, log_sequences_str, log_ground_truth, log_format_score, log_correctness_score], axis=1))

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(format_score_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(format_score_tensor[i].item())

        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_format_score/{data_source}'] = np.mean(rewards)

        data_source_reward = {}
        for i in range(correctness_score_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(correctness_score_tensor[i].item())

        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_correctness_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.loss_type == 'with_g':
            assert self.config.data.filter_time >= 1
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.loss_type == 'without_g':
            assert self.config.data.filter_time == 1
            self.use_critic = False
            self.config.algorithm.alpha = 1
            self.config.algorithm.alpha_decay_coef = 0
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
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

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _upload_checkpoint(self):
        if self.config.trainer.default_hub_dir is not None:
            actor_hub_path = self.config.trainer.default_hub_dir + '_actor'
            actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
            # init model and tokenizer
            temp_model = AutoModelForCausalLM.from_pretrained(actor_local_path, torch_dtype=torch.bfloat16)
            temp_tokenizer = AutoTokenizer.from_pretrained(actor_local_path)
            temp_model = temp_model.cpu()
            temp_model.push_to_hub(actor_hub_path)
            temp_tokenizer.push_to_hub(actor_hub_path)
            del temp_model
            del temp_tokenizer
            if self.use_critic:
                critic_hub_path = self.config.trainer.default_hub_dir + '_critic'
                critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
                # init model and tokenizer
                temp_model = AutoModelForTokenClassification.from_pretrained(critic_local_path, torch_dtype=torch.bfloat16)
                temp_tokenizer = AutoTokenizer.from_pretrained(critic_local_path)
                temp_model = temp_model.cpu()
                temp_model.push_to_hub(critic_hub_path)
                temp_tokenizer.push_to_hub(critic_hub_path)
                del temp_model
                del temp_tokenizer

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
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

    def fit(self):
        """
        The training loop of Explore.
        The driver process only need to call the compute functions of the worker group through RPC to construct the Explore dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        time_id = int(time.time())
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=f"{time_id}_{self.config.trainer.experiment_name}",
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

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

        # assert replay buffer and number of generations
        if self.config.algorithm.load_from_replay == False and self.config.actor_rollout_ref.rollout.n == 1:
            print("setting 1: not loading from replay buffer and not doing multiple generations")
        elif self.config.algorithm.load_from_replay == False and self.config.actor_rollout_ref.rollout.n > 1:
            print("setting 2: not loading from replay buffer and doing multiple generations")
        elif self.config.algorithm.load_from_replay == True and self.config.actor_rollout_ref.rollout.n == 2:
            print("setting 3: loading from replay buffer and doing multiple generations")
        else:
            raise NotImplementedError

        # initiaze alpha
        alpha_step = self.global_steps - self.config.algorithm.alpha_start_step - 1
        iteration_num_step = self.config.algorithm.alpha_alternate_on_step + self.config.algorithm.alpha_alternate_off_step
        if self.config.algorithm.alpha_start_step < self.global_steps and \
           self.global_steps < self.total_training_steps - self.config.algorithm.alpha_end_step and \
           alpha_step % iteration_num_step < self.config.algorithm.alpha_alternate_on_step:
            alpha = 1 / self.config.algorithm.alpha / (self.global_steps ** self.config.algorithm.alpha_decay_coef)
        else:
            alpha = 0

        replay_buffer = {}
        replay_buffer_records = defaultdict(list)
        replay_buffer_g_x = {}
        for epoch in range(int(self.config.trainer.total_epochs * self.config.data.filter_time)):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                with _timer('step', timing_raw):

                    if self.config.data.filter_time > 1:
                        old_values = self.critic_wg.filter_based_on_value(batch).batch['values'].squeeze()
                        selected = torch.argsort(old_values, descending=False)[:self.config.data.train_batch_size]
                        batch.batch = batch.batch[selected]
                        for key in batch.non_tensor_batch.keys():
                            batch.non_tensor_batch[key] = batch.non_tensor_batch[key][selected]

                    # generate a batch
                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids']) # pop those keys for generation
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # print(f"union batch.batch: {batch.batch}")
                    # print(f"union batch.batch['prompts'][0]: {batch.batch['prompts'][0]}")
                    # print(f"union batch.batch['responses'][0]: {batch.batch['responses'][0]}")
                    # print(f"union batch.batch['attention_mask'][0]: {batch.batch['attention_mask'][0]}")
                    # print(f"union batch.batch['input_ids'][0]: {batch.batch['input_ids'][0]}")
                    # print(f"union batch.batch['input_ids'][0] * batch.batch['attention_mask'][0]: {batch.batch['input_ids'][0] * batch.batch['attention_mask'][0]}")
                    # print(f"union batch.batch['position_ids'][0]: {batch.batch['position_ids'][0]}")
                    # print(f"union batch.batch['position_ids'][0] * batch.batch['attention_mask'][0]: {batch.batch['position_ids'][0] * batch.batch['attention_mask'][0]}")
                    if self.config.data.mixing > 0:

                        evals = [batch.non_tensor_batch[f'eval_{i}'] for i in range(self.config.data.num_gen_to_use)]
                        evals = np.asarray(evals, dtype=float)

                        # print(f"evals: {evals}")
                        # print(f"evals shape: {evals.shape}")

                        num_to_replace = int(self.config.data.mixing * len(batch.batch))
                        if num_to_replace > np.max(evals, axis=0).sum():
                            num_to_replace = np.max(evals, axis=0).sum()

                        # print(f'num_to_replace: {num_to_replace}')
                        # print(f'np.max(evals, axis=0): {np.max(evals, axis=0)}')
                        # print(f'np.max(evals, axis=0).sum(): {np.max(evals, axis=0).sum()}')

                        index_to_replace = np.random.choice(len(batch.batch), num_to_replace, replace=False)
                        index_to_replace = np.random.choice(len(batch.batch), num_to_replace, replace=False, p=np.max(evals, axis=0) / np.max(evals, axis=0).sum())

                        # print(f'index_to_replace: {index_to_replace}')

                        for i in index_to_replace:
                            # print(f"evals[:, i]: {evals[:, i]}")
                            response_idx = np.random.choice(self.config.data.num_gen_to_use, 1, p=evals[:, i] / evals[:, i].sum())
                            # print(f"response_idx: {response_idx}")
                            response = batch.non_tensor_batch[f'response_{response_idx[0]}'][i] + "<|endoftext|>"
                            # print(f"response: {response}")
                            tokenized_response = self.tokenizer(response, return_tensors='pt', padding='max_length', truncation=True, max_length=self.config.data.max_response_length)
                            # print(f"tokenized_response: {tokenized_response}")

                            batch.batch['responses'][i] = tokenized_response.input_ids[0]
                            batch.batch['attention_mask'][i, -self.config.data.max_response_length:] = tokenized_response.attention_mask[0]
                            batch.batch['input_ids'][i, -self.config.data.max_response_length:] = tokenized_response.input_ids[0]

                            # print(f"batch.batch['responses'][i]: {batch.batch['responses'][i]}")
                            # print(f"batch.batch['attention_mask'][i]: {batch.batch['attention_mask'][i]}")
                            # print(f"batch.batch['input_ids'][i]: {batch.batch['input_ids'][i]}")
                            # print(f"batch.batch['input_ids'][i] * batch.batch['attention_mask'][i]: {batch.batch['input_ids'][i] * batch.batch['attention_mask'][i]}")
                            # print(self.tokenizer.decode(batch.batch['input_ids'][i], skip_special_tokens=True))

                    # compute scores. Support both model and function-based.
                    # We first compute the scores using reward model. Then, we call reward_fn to combine
                    # the results from reward model and rule-based results.
                    if self.use_rm:
                        # we first compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    # we combine with rule-based rm
                    format_score_tensor, correctness_score_tensor, _ = self.reward_fn(batch)
                    reward_tensor = format_score_tensor + correctness_score_tensor
                    batch.batch['token_level_scores'] = reward_tensor
                    batch.batch['token_level_format_scores'] = format_score_tensor
                    batch.batch['token_level_correctness_scores'] = correctness_score_tensor

                    # =====================================================================================================================

                    # add to replay buffer
                    buffer_metric = {
                        'buffer/both_correct': 0,
                        'buffer/both_incorrect': 0,
                        'buffer/buffer_incorrect_sampled_correct': 0,
                        'buffer/buffer_correct_sampled_incorrect': 0,
                        'buffer/add_to_batch': 0,
                    }
                    prompt_text = batch.non_tensor_batch['extra_info']
                    for i in range(len(prompt_text)):

                        # check correctness
                        batch_correct = batch.batch['token_level_scores'][i].sum().item() >= 0.5
                        if prompt_text[i]['index'] in replay_buffer:
                            replay_buffer_correct = replay_buffer[prompt_text[i]['index']].batch['token_level_scores'].sum().item() >= 0.5
                        else:
                            replay_buffer_correct = False
                        replay_buffer_records[prompt_text[i]['index']].append(batch_correct)

                        # add to replay buffer if not in buffer
                        if prompt_text[i]['index'] not in replay_buffer:
                            replay_buffer[prompt_text[i]['index']] = copy.deepcopy(batch[i])

                        # add to replay buffer if both correct
                        elif batch_correct == True and replay_buffer_correct == True:
                            buffer_metric['buffer/both_correct'] += 1
                            replay_buffer[prompt_text[i]['index']] = copy.deepcopy(batch[i])

                        # add to replay buffer if both incorrect
                        elif batch_correct == False and replay_buffer_correct == False:
                            buffer_metric['buffer/both_incorrect'] += 1
                            replay_buffer[prompt_text[i]['index']] = copy.deepcopy(batch[i])

                        # add to replay buffer if buffer is incorrect but sampled is correct
                        elif batch_correct == True and replay_buffer_correct == False:
                            buffer_metric['buffer/buffer_incorrect_sampled_correct'] += 1
                            replay_buffer[prompt_text[i]['index']] = copy.deepcopy(batch[i])

                        # add to the current batch if buffer is correct but sampled is incorrect
                        elif batch_correct == False and replay_buffer_correct == True:
                            buffer_metric['buffer/buffer_correct_sampled_incorrect'] += 1
                            if self.config.algorithm.load_from_replay == True:
                                
                                # gather relative index
                                start_i = (i // self.config.actor_rollout_ref.rollout.n) * self.config.actor_rollout_ref.rollout.n
                                end_i = start_i + self.config.actor_rollout_ref.rollout.n

                                # add the batch if all are incorrect
                                if torch.all(batch.batch['token_level_scores'][start_i:end_i].sum(-1) < 0.5).item():
                                    buffer_metric['buffer/add_to_batch'] += 1
                                    for key in batch.batch.keys():
                                        batch.batch[key][i] = replay_buffer[prompt_text[i]['index']].batch[key]
                                    for key in batch.non_tensor_batch.keys():
                                        batch.non_tensor_batch[key][i] = replay_buffer[prompt_text[i]['index']].non_tensor_batch[key]

                    # log buffer metrics
                    metrics.update(buffer_metric)

                    # compute the leave one out r
                    if self.config.actor_rollout_ref.rollout.n == 1:
                        batch.batch["avg_reward"] = torch.zeros_like(batch.batch["token_level_scores"]).sum(-1)
                    else:
                        avg_reward = batch.batch["token_level_scores"].sum(-1)
                        avg_reward = avg_reward.view(-1, self.config.actor_rollout_ref.rollout.n)
                        avg_reward = avg_reward.sum(-1).repeat_interleave(self.config.actor_rollout_ref.rollout.n)
                        batch.batch["avg_reward"] = (avg_reward - batch.batch["token_level_scores"].sum(-1)) / (self.config.actor_rollout_ref.rollout.n - 1)

                    # =====================================================================================================================

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    batch.meta_info['alpha'] = alpha

                    # compute old logprob
                    old_log_prob = self.actor_rollout_wg.compute_old_log_prob(batch)
                    batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute rewards. apply_kl_penalty if available
                    if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        batch, kl_metrics = apply_kl_penalty(batch,
                                                                kl_ctrl=self.kl_ctrl,
                                                                kl_penalty=self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)

                            # nomalize values
                            if self.config.algorithm.normalize_g:
                                values = values.batch['values']
                                values[:, 0] = (values[:, 0] - values[:, 0].mean()) / (values[:, 0].std() + 1e-5)
                                values = DataProto.from_dict(tensors={'values': values})
                            batch = batch.union(values)

                    # add computed g(x) to the buffer
                    if self.use_critic:
                        for i in range(len(prompt_text)):
                            replay_buffer_g_x[prompt_text[i]['index']] = batch.batch['values'][i, 0].item()
                    else:
                        for i in range(len(prompt_text)):
                            replay_buffer_g_x[prompt_text[i]['index']] = 0

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        # push_buffer_to_hub(replay_buffer, replay_buffer_records, replay_buffer_g_x, time_id)
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                            self._upload_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                logger.log(data={'critic/alpha': alpha}, step=self.global_steps)

                # update global_steps and alpha
                self.global_steps += 1
                alpha_step = self.global_steps - self.config.algorithm.alpha_start_step - 1
                iteration_num_step = self.config.algorithm.alpha_alternate_on_step + self.config.algorithm.alpha_alternate_off_step
                if self.config.algorithm.alpha_start_step < self.global_steps and \
                self.global_steps < self.total_training_steps - self.config.algorithm.alpha_end_step and \
                alpha_step % iteration_num_step < self.config.algorithm.alpha_alternate_on_step:
                    alpha = 1 / self.config.algorithm.alpha / (self.global_steps ** self.config.algorithm.alpha_decay_coef)
                else:
                    alpha = 0

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        # push_buffer_to_hub(replay_buffer, replay_buffer_records, replay_buffer_g_x, time_id)
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)

                    # make sure wandb finishes logging
                    if 'wandb' in self.config.trainer.logger:
                        logger.logger['wandb'].finish()

                    return
