from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import BloomModel, GPT2Model, OPTModel
from transformers.modeling_outputs import ModelOutput

from .pairwise_loss import PairWiseLoss


@dataclass
class RewardModelOutput(ModelOutput):
    """Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
    """

    loss: Optional[torch.FloatTensor] = None
    rewards_chosen: torch.FloatTensor = None
    rewards_rejected: torch.FloatTensor = None


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
        return_dict=None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            input_ids (torch.LongTensor): Input tensor of token ids.
            attention_mask (Optional[torch.Tensor]): Input tensor of attention masks.

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
            output = (loss, ) + rewards_chosen + rewards_rejected
            return output

        return RewardModelOutput(loss=loss,
                                 rewards_chosen=rewards_chosen,
                                 rewards_rejected=rewards_rejected)
