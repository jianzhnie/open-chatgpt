from typing import Any, Dict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class OpenWebMathDataset(Dataset):
    """

    Args:
        data_path (str): Path to the training data.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        split (str): The split to use from the training data.
        max_length (int): The maximum length of the input sequences (default: 550).
    """

    def __init__(self,
                 data_path: str,
                 split: str,
                 cache_dir: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024) -> None:

        self.dataset = load_dataset(data_path,
                                    split=split,
                                    cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a dictionary containing the input_ids, attention_mask, and
        labels for the given index.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            A dictionary containing the input_ids, attention_mask, and labels.
        """
        sample = self.dataset[idx]
        text = sample['text']  # Extract the 'text' field from the dataset

        encodings_input = self.tokenizer(text,
                                         truncation=True,
                                         max_length=self.max_length,
                                         padding='max_length')

        encodings_input = {
            key: torch.tensor(val)
            for key, val in encodings_input.items()
        }
        return encodings_input
