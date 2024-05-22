from abc import abstractmethod
from typing import Callable, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .data_types import GeneralElement


class BasePipeline(Dataset):

    def __init__(self, path: str = 'dataset'):
        super().__init__()

    @abstractmethod
    def __getitem__(self, index: int) -> GeneralElement:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
        prep_fn: Callable = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a dataloader for the pipeline.

        :param prep_fn: Typically a tokenizer. Applied to GeneralElement after collation.
        """
        pass


class PromptPipeline(BasePipeline):
    """Tokenizes prompts, unless they are already tokenized, and truncates them
    to `max_prompt_length` from the right."""

    def __init__(self, prompts: List[str], max_prompt_length: int,
                 tokenizer: PreTrainedTokenizer):
        super().__init__()

        model_inputs = tokenizer(prompts,
                                 truncation=True,
                                 padding=False,
                                 max_length=max_prompt_length,
                                 add_special_tokens=False)

        prompts_tokens = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        self.tokenizer = tokenizer
        self.prompts = [{
            'input_ids': tokens,
            'attention_mask': mask
        } for tokens, mask in zip(prompts_tokens, attention_mask)]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        collate_fn = DataCollatorWithPadding(
            self.tokenizer) if self.tokenizer else torch.vstack
        return DataLoader(self,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle)
