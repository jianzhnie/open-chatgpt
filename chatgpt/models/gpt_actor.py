from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


class GPTActor():
    """GPT Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
    """
    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False) -> None:
        super().__init__()
        if pretrained is not None:
            self.model = GPT2LMHeadModel.from_pretrained(pretrained)
        elif config is not None:
            self.model = GPT2LMHeadModel(config)
        else:
            self.model = GPT2LMHeadModel(GPT2Config())
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 return_action_mask: bool = True,
                 **kwargs):

        pass

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns action log probs."""
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits']

        logits = logits[:, :-1, :]
        labels = sequences[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
        log_probs = log_probs_labels[:, -num_actions]
        return log_probs
