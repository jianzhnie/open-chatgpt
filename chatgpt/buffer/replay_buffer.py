import json
import random
from collections import namedtuple
from typing import Deque, List, Tuple

import torch
from torch.utils.data import Dataset

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
