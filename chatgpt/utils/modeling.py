from typing import Any, Dict, List, MutableMapping, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al.

    "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """
    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.

        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2,
                                     0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁


class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.

        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        pass


def get_global_statistics(xs: torch.Tensor) -> Tuple[float, float, int]:
    """Computes element-wise mean and variance of the tensor across
    processes."""
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean)**2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM)
    global_var = sum_var / count
    return global_mean, global_var, count


def whiten(xs: torch.Tensor,
           shift_mean=True,
           distributed=True) -> torch.Tensor:
    """Whitens values."""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def flatten_dict(
    d: Union[dict, MutableMapping],
    parent_key: str = '',
    sep: str = '/',
) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_tensor_stats(xs: torch.Tensor, mask: torch.Tensor, n: int):
    mean = (xs * mask).sum() / n
    return dict(
        mean=mean,
        min=torch.where(mask.bool(), xs, np.inf).min(),
        max=torch.where(mask.bool(), xs, -np.inf).max(),
        std=torch.sqrt(((xs - mean) * mask).pow(2).sum() / n),
    )


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels.

    These are calculated from the logits.
    """
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs,
                                   dim=-1,
                                   index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)
