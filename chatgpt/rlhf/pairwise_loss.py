import torch
import torch.nn as nn


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Ranking Tasks
    
    This PyTorch module computes a pairwise loss for ranking tasks where the goal is to compare 
    two inputs and determine which one is "better" than the other. Given two input tensors: 
    `chosen_reward` and `reject_reward`, which should contain reward values for the "chosen" 
    and "rejected" options, respectively, this module computes the probability of the chosen 
    option being "better" than the rejected option using a sigmoid function, and then takes 
    the negative logarithm of that probability to get the loss. The loss is then averaged over 
    the batch dimension and returned as a scalar tensor. Note that this module assumes that 
    higher reward values indicate better options.
    """
    def __init__(self):
        super(PairWiseLoss, self).__init__()

    def forward(self, chosen_reward: torch.Tensor,
                reject_reward: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise loss
        
        Args:
        - chosen_reward: A tensor of shape (batch_size,) containing reward values for the chosen option
        - reject_reward: A tensor of shape (batch_size,) containing reward values for the rejected option
        
        Returns:
        - loss: A scalar tensor containing the computed pairwise loss
        """

        # Compute probability of the chosen option being better than the rejected option
        probs = torch.sigmoid(chosen_reward - reject_reward)

        # Take the negative logarithm of the probability to get the loss
        log_probs = torch.log(probs)
        loss = -log_probs.mean()

        return loss
