from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import BloomModel, GPT2Model, OPTModel
from transformers.modeling_outputs import ModelOutput

from .pairwise_loss import PairWiseLoss


@dataclass
class RewardModelOutput(ModelOutput):
    """A class representing the output of a reward-based machine learning
    model.

    Attributes:
        loss (`Optional[torch.FloatTensor]`, optional): The classification or regression loss of the model.
        rewards_chosen (`torch.FloatTensor`): The rewards for the chosen actions.
        rewards_rejected (`torch.FloatTensor`): The rewards for the rejected actions.
    """

    # Define class attributes with type annotations and default values
    loss: Optional[torch.FloatTensor] = None
    rewards_chosen: torch.FloatTensor = None
    rewards_rejected: torch.FloatTensor = None


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


class MeanPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        mean_token_tensor = torch.mean(hidden_states)
        pooled_output = self.dense(mean_token_tensor)
        return pooled_output


class RewardModel(nn.Module):
    """GPT Reward model.

    Args:
        model (str): Model name: 'opt', 'gpt2' or 'bloom'
        pretrained (str): Pretrained model name or path.
    """
    def __init__(self, model: str = '', pretrained: str = 'openai-gpt'):
        super().__init__()
        # Instantiate model based on input string
        if model == 'opt':
            self.model = OPTModel.from_pretrained(pretrained)
        elif model == 'gpt2':
            self.model = GPT2Model.from_pretrained(pretrained)
        elif model == 'bloom':
            self.model = BloomModel.from_pretrained(pretrained)
        else:
            # If the model string is invalid, raise an error
            raise ValueError(
                "Invalid model name. Choose 'opt', 'gpt2' or 'bloom'.")

        # Get the model's config and create a value head
        self.config = self.model.config
        self.value_head = nn.Linear(self.config.word_embed_proj_dim, 1)
        self.loss_fn = PairWiseLoss()

    def forward(
        self,
        chosen_input_ids: torch.LongTensor,
        rejected_input_ids: torch.LongTensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict=None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            chosen_input_ids (torch.LongTensor): Tensor of token ids for the chosen sequence.
            rejected_input_ids (torch.LongTensor): Tensor of token ids for the rejected sequence.
            chosen_attention_mask (Optional[torch.Tensor]): Tensor of attention masks for the chosen sequence.
            rejected_attention_mask (Optional[torch.Tensor]): Tensor of attention masks for the rejected sequence.
            return_dict (Optional[bool]): Whether to return a dictionary of outputs instead of a tuple.

        Returns:
            torch.Tensor: Output tensor of value estimates.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Get the model's outputs and extract the last hidden state
        outputs_chosen = self.model(input_ids=chosen_input_ids,
                                    attention_mask=chosen_attention_mask,
                                    return_dict=return_dict)
        last_hidden_states_chosen = outputs_chosen['last_hidden_state']

        # Calculate the values and return the mean value for each sequence
        values_chosen = self.value_head(last_hidden_states_chosen)[:, :-1]
        rewards_chosen = values_chosen.mean(dim=1).squeeze(1)

        # Get the model's outputs and extract the last hidden state
        outputs_rejected = self.model(input_ids=rejected_input_ids,
                                      attention_mask=rejected_attention_mask,
                                      return_dict=return_dict)
        last_hidden_states_rejected = outputs_rejected['last_hidden_state']

        # Calculate the values and return the mean value for each sequence

        values_rejected = self.value_head(last_hidden_states_rejected)[:, :-1]
        rewards_rejected = values_rejected.mean(dim=1).squeeze(1)

        loss = self.loss_fn(rewards_chosen, rewards_rejected)

        if not return_dict:
            output = (loss, rewards_chosen, rewards_rejected)
            return output

        return RewardModelOutput(loss=loss,
                                 rewards_chosen=rewards_chosen,
                                 rewards_rejected=rewards_rejected)
