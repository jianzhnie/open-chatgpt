import json
import random
from abc import ABC
from collections import namedtuple
from typing import Deque, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from chatgpt.buffer.utils import (BufferItem, Experience,
                                  make_experience_batch,
                                  split_experience_batch)
from chatgpt.rlhf.actor_critic import ActorModel
from chatgpt.utils.modeling import compute_reward

# structure to store the data for each experience
Memory = namedtuple(
    'Memory',
    [
        'states_actor',
        'actions',
        'values',
        'rewards',
        'actions_log_probs',
        'sequences_actor',
        'sequences_mask_actor',
        'sequences_critic',
        'sequences_mask_critic',
        'action_len_actor',
        'action_len_critic',
    ],
)

DsMemory = namedtuple(
    'ds_memory',
    [
        'prompts',
        'logprobs',
        'ref_logprobs',
        'value',
        'rewards',
        'input_ids',
        'attention_mask',
    ],
)


class DsExperienceDataset(Dataset):
    """Dataset to train the actor-critic models."""

    def __init__(self, memories: Deque[Memory]) -> None:
        super().__init__()
        self.data = memories
        print(self.data)

    def __len__(self, ) -> int:
        return self.data[0].shape[0]

    def __getitem__(self, idx) -> Tuple:
        # return the idx-th memory element as a tuple of tensors on the device
        # item = (
        #     self.data[idx].prompts,
        #     self.data[idx].logprobs,
        #     self.data[idx].ref_logprobs,
        #     self.data[idx].rewards,
        #     self.data[idx].input_ids,
        #     self.data[idx].attention_mask,
        #     self.data[idx].value,
        # )
        item = tuple(map(lambda t: t[idx], self.data))
        return item


class ExperienceDataset(Dataset):
    """Dataset to train the actor-critic models."""

    def __init__(self, memories: Deque[Memory]) -> None:
        super().__init__()
        self.data = list(memories)

    def __len__(self, ) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple:
        # return the idx-th memory element as a tuple of tensors on the device
        item = (
            self.data[idx].states_actor,
            self.data[idx].actions,
            self.data[idx].values,
            self.data[idx].rewards,
            self.data[idx].actions_log_probs,
            self.data[idx].sequences_actor,
            self.data[idx].sequences_mask_actor,
            self.data[idx].sequences_critic,
            self.data[idx].sequences_mask_critic,
            int(self.data[idx].action_len_actor),
            int(self.data[idx].action_len_critic),
        )
        return item


class ExperienceMaker(ABC):

    def __init__(self,
                 actor: ActorModel,
                 critic: nn.Module,
                 reward_model: nn.Module,
                 initial_model: ActorModel,
                 kl_coef: float = 0.1) -> None:
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.kl_coef = kl_coef

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor,
                        **generate_kwargs) -> Experience:

        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        print('4.7')
        sequences, attention_mask, action_mask = self.actor.generate(
            input_ids, return_action_mask=True, **generate_kwargs)
        print('4.8')

        action_log_probs = self.actor(sequences, attention_mask)
        print('4.9')
        base_action_log_probs = self.initial_model(sequences, attention_mask)
        print('4.10')
        value = self.critic(sequences, action_mask, attention_mask)
        print('4.11')
        r = self.reward_model(sequences, attention_mask)
        print('4.12')
        reward = compute_reward(r,
                                self.kl_coef,
                                action_log_probs,
                                base_action_log_probs,
                                action_mask=action_mask)

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward,
                          advantage, attention_mask, action_mask)


class ReplayBuffer(ABC):

    def __init__(self,
                 max_len: int = 10000,
                 sample_batch_size: int = 8,
                 device='cpu') -> None:
        super().__init__()
        self.max_len = max_len
        self.sample_batch_size = sample_batch_size
        self.device = device

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.max_len > 0:
            samples_to_remove = len(self.items) - self.max_len
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch)
        return experience
