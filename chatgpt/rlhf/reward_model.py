import torch
from torch import nn
from transformers import OPTModel
from transformers import GPT2Model
from transformers import BloomModel
from typing import Optional


class RewardModel(nn.Module):
    """
    GPT Reward model.

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

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.LongTensor): Input tensor of token ids.
            attention_mask (Optional[torch.Tensor]): Input tensor of attention masks.

        Returns:
            torch.Tensor: Output tensor of value estimates.
        """
        # Get the model's outputs and extract the last hidden state
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']

        # Calculate the values and return the mean value for each sequence
        values = self.value_head(last_hidden_states)[:, :-1]
        value = values.mean(dim=1).squeeze(1)

        return value
