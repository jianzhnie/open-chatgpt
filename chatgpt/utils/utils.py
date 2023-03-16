import random

import numpy as np
import torch
import torch.nn.functional as F


class LengthSampler:
    """
    Samples a length
    """
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)
