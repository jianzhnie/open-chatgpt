from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset


class PairwiseDataset(Dataset):
    """Dataset class for pairwise ranking tasks.

    Args:
        data_path: Path to the dataset.
        tokenizer: The tokenizer used to encode the input text.
        max_length: Maximum sequence length for the encoded inputs.
    """
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer,
                 split: str, max_length: int):

        self.pairs = self.create_comparison_dataset(data_path, split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError(
                f'Index {idx} out of range for TLDRDataset with length {len(self)}'
            )
        pair = self.pairs[idx]
        chosen_example, rejected_example = pair['chosen'], pair['rejected']

        chosen_encodings_dict = self.tokenizer(
            chosen_example,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        rejected_encodings_dict = self.tokenizer(
            rejected_example,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        encodings_input = {}
        encodings_input['chosen_input_ids'] = chosen_encodings_dict[
            'input_ids']
        encodings_input['chosen_attention_mask'] = chosen_encodings_dict[
            'attention_mask']
        encodings_input['rejected_input_ids'] = rejected_encodings_dict[
            'input_ids']
        encodings_input['rejected_attention_mask'] = rejected_encodings_dict[
            'attention_mask']

        encodings_input = {
            key: torch.tensor(val)
            for key, val in encodings_input.items()
        }

        return encodings_input

    def create_comparison_dataset(self, path: str, split: str = "train"):
        dataset = load_dataset(path, split=split)
        pairs = []
        for prompt, chosen_summary, rejected_summary in zip(
                dataset['prompt'], dataset['chosen'], dataset['rejected']):
            pair = {}
            if chosen_summary == rejected_summary:
                continue
            if len(chosen_summary.split()) < 5 or len(
                    rejected_summary.split()) < 5:
                continue

            pair["chosen"] = prompt + "\n" + chosen_summary + '<|endoftext|>'
            pair["rejected"] = prompt + "\n" + rejected_summary + '<|endoftext|>'
            pairs.append(pair)
        return pairs


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
