from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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
                 max_length: int = 512) -> None:

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
        encodings_labels = self.tokenizer(summary_txt,
                                          truncation=True,
                                          max_length=self.max_length,
                                          padding='max_length')

        encodings_input['labels'] = encodings_labels['input_ids']

        encodings_input = {
            key: torch.tensor(val)
            for key, val in encodings_input.items()
        }
        return encodings_input


class HFSummaryQuality(Dataset):
    def __init__(
        self,
        data_path: str = 'openai/summarize_from_feedback',
        split: str = 'validation',
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 300,
    ):
        """A PyTorch dataset for evaluating the quality of summaries generated
        from feedback.

        Args:
            data_path (str, optional): The path to the dataset. Defaults to "openai/summarize_from_feedback".
            split (str, optional): The dataset split to use (validation or test). Defaults to "validation".
            tokenizer (PreTrainedTokenizer, optional): The tokenizer to use. Defaults to None.
            max_length (int, optional): The maximum length of the summary. Defaults to 300.
        """
        super().__init__()

        assert split in ('validation', 'test')

        # Load the dataset
        dataset = load_dataset(data_path, 'axis')[split]

        self.max_length = max_length

        # Store the contexts, responses, and labels
        self.contexts: List[str] = []
        self.responses: List[str] = []
        self.labels: List[Dict[str, Optional[float]]] = []

        # Compute the mean score for each axis
        mean_scores = defaultdict(list)
        for data in dataset:
            if 'article' in data['info'] and data['info'][
                    'article'] is not None:
                context = data['info']['article']
            elif 'post' in data['info']:
                context = data['info']['post']
            self.contexts.append(context)

            response = data['summary']['text']
            self.responses.append(response)
            self.labels.append(data['summary']['axes'])

            for axis, score in data['summary']['axes'].items():
                if score is not None:
                    mean_scores[axis].append(score)

        # Create a mapping from axis names to indices
        self.label2idx = {
            key: idx
            for idx, key in enumerate(mean_scores.keys())
        }

        # Compute the mean score for each axis
        self.label2mean = {
            key: np.mean(scores)
            for key, scores in mean_scores.items()
        }

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Returns the number of summaries in the dataset.

        Returns:
            int: The number of summaries in the dataset.
        """
        return len(self.responses)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, Union[torch.Tensor, Any]], torch.Tensor]:
        """Retrieve an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            A tuple containing:
                - A dictionary with the keys "input_ids", "attention_mask", and "labels".
                The values associated with each key are PyTorch tensors representing the encoded context,
                    response, and labels, respectively.
                - A NumPy array containing the labels for the summary quality on different axes.
        """
        # Get the context and response for the given index
        context = self.contexts[index]
        response = self.responses[index]

        # Create a dictionary containing the encoded inputs and labels
        encoded = self.tokenizer(context,
                                 response,
                                 truncation=True,
                                 max_length=self.max_length,
                                 padding='max_length')

        # Get the labels for the summary quality on different axes
        labels = np.zeros(len(self.label2idx))
        for key, score in self.labels[index].items():
            if score is not None:
                labels[self.label2idx[
                    key]] = score / 10  # Normalize scores to be between 0 and 1

        encoded['labels'] = torch.tensor(labels, dtype=torch.float32)
        # Return the encoded inputs and labels as a tuple
        return encoded
