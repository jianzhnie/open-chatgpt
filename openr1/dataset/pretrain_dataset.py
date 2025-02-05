from typing import Dict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class PretrainDataset(Dataset):
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
                 cache_dir: str = None,
                 tokenizer: PreTrainedTokenizer = None,
                 max_length: int = 1024) -> None:

        self.dataset = load_dataset(data_path,
                                    split=split,
                                    cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieves and preprocesses a single data sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - input_ids: Tokenized and padded input sequence
                - attention_mask: Mask indicating non-padded tokens
                - token_type_ids: (if using BERT-like models)
        """
        # Get text sample from dataset
        sample = self.dataset[idx]
        text: str = sample['text']

        # Tokenize and pad the text
        encodings_input = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'  # Return PyTorch tensors directly
        )

        # Remove the batch dimension added by return_tensors='pt'
        return {key: val.squeeze(0) for key, val in encodings_input.items()}
