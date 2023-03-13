from typing import Any, Dict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TLDRDataset(Dataset):
    """A PyTorch Dataset for TLDR training data.

    Args:
        data_path (str): Path to the training data.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        split (str): The split to use from the training data.
        max_length (int): The maximum length of the input sequences (default: 550).
    """

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 split: str,
                 max_length: int = 550) -> None:

        dataset = load_dataset(data_path, split=split)
        self.post_list = [(sample['prompt'], sample['label'])
                          for sample in dataset]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.post_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a dictionary containing the input_ids, attention_mask, and
        labels for the given index.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            A dictionary containing the input_ids, attention_mask, and labels.
        """
        if idx < 0 or idx >= len(self.post_list):
            raise IndexError(
                f'Index {idx} out of range for TLDRDataset with length {len(self)}'
            )

        input_txt, summary_txt = self.post_list[idx]
        encodings_input = self.tokenizer(input_txt,
                                         truncation=True,
                                         max_length=self.max_length,
                                         padding='max_length')
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            encodings_labels = self.tokenizer(summary_txt,
                                              truncation=True,
                                              max_length=self.max_length,
                                              padding='max_length')

        encodings_input['labels'] = encodings_labels['input_ids']
        encodings_input['summary_attention_mask'] = encodings_labels[
            'attention_mask']
        encodings_input = {
            key: torch.tensor(val)
            for key, val in encodings_input.items()
        }

        return encodings_input