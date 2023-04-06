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
        'states',
        'actions',
        'sequences',
        'values',
        'rewards',
        'actions_log_probs',
        'sequences_mask',
    ],
)


class ExperienceDataset(Dataset):
    """Dataset to train the actor-critic models."""
    def __init__(self, memories: Deque[Memory], device: torch.device) -> None:
        super().__init__()
        self.data = list(memories)
        self.device = device

    def __len__(self, ) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple:
        # return the idx-th memory element as a tuple of tensors on the device
        item = (
            self.data[idx].states.to(self.device),
            self.data[idx].actions.to(self.device),
            self.data[idx].sequences.to(self.device),
            self.data[idx].values.to(self.device),
            self.data[idx].rewards.to(self.device),
            self.data[idx].actions_log_probs.to(self.device),
            self.data[idx].sequences_mask.to(self.device),
        )
        return item


class ExamplesSampler:
    """Store the prompt to be sampled to generate the examples
    read a json file with the following format:
    [
        {
            "user_input" : "",
        } ,
        ...
    ]
    Where:
        user_input: is the input of the user or directly the input of the user
            with the memory preappended (i.e. user_input + memory)
    """
    def __init__(
        self,
        path: str,
    ) -> None:
        self.path = path
        with open(path, 'r') as f:
            data = json.load(f)
        self.data = [d['user_input'] for d in data]

    def sample(self, n: int) -> List:
        """Sample n examples from the data.

        Args:
            n (int): Number of examples to sample
        """
        return random.sample(self.data, n)


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

        sequences, attention_mask, action_mask = self.actor.generate(
            input_ids, return_action_mask=True, **generate_kwargs)
        num_actions = action_mask.size(1)

        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        base_action_log_probs = self.initial_model(sequences, num_actions,
                                                   attention_mask)
        value = self.critic(sequences, action_mask, attention_mask)
        r = self.reward_model(sequences, attention_mask)
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
