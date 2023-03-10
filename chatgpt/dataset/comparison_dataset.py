from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class PairwiseDataset(Dataset):
    """Dataset class for pairwise ranking tasks.

    Args:
        pairs: List of dictionaries containing 'positive' and 'negative' keys.
        tokenizer: The tokenizer used to encode the input text.
        max_length: Maximum sequence length for the encoded inputs.
    """
    def __init__(self, pairs: List[Dict[str, str]],
                 tokenizer: PreTrainedTokenizer, max_length: int):
        self.positive_input_ids = []
        self.positive_attn_masks = []
        self.negative_input_ids = []
        self.negative_attn_masks = []

        for pair in pairs:
            positive_example, negative_example = pair['positive'], pair[
                'negative']
            positive_encodings_dict = tokenizer(
                positive_example,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
            )
            negative_encodings_dict = tokenizer(
                negative_example,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
            )
            if not positive_encodings_dict or not negative_encodings_dict:
                raise ValueError('Empty encoding dictionary.')
            self.positive_input_ids.append(
                positive_encodings_dict['input_ids'])
            self.positive_attn_masks.append(
                positive_encodings_dict['attention_mask'])
            self.negative_input_ids.append(
                negative_encodings_dict['input_ids'])
            self.negative_attn_masks.append(
                negative_encodings_dict['attention_mask'])

    def __len__(self) -> int:
        return len(self.positive_input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'positive_input_ids': torch.tensor(self.positive_input_ids[idx]),
            'positive_attention_mask':
            torch.tensor(self.positive_attn_masks[idx]),
            'negative_input_ids': torch.tensor(self.negative_input_ids[idx]),
            'negative_attention_mask':
            torch.tensor(self.negative_attn_masks[idx])
        }


class RewardDataCollator:
    """A data collator for binary classification tasks.

    Args:
        data (list): A list of tuples containing tensors of the same size.
    Returns:
        A dictionary containing concatenated input tensors and labels.
    """
    def __call__(self,
                 data: List[Tuple[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Check that the input data is a list of tuples containing tensors of the same size
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError('Input data must be a non-empty list.')

        # Concatenate the input tensors using default_collate
        input_ids = torch.cat([x[0] for x in data] + [x[2] for x in data])
        attention_masks = torch.cat([x[1]
                                     for x in data] + [x[3] for x in data])
        labels = torch.cat([torch.zeros(len(data)), torch.ones(len(data))])

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }

        return batch
