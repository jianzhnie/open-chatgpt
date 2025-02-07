from typing import Dict

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from openr1.utils.logger_utils import get_logger

logger = get_logger('openr1')


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Attributes:
        PROMPT_DICT (dict): A dictionary containing prompts for the model to complete.
        IGNORE_INDEX (int): A value to replace tokens corresponding to the source in the labels tensor.

    Methods:
        __init__(self, data_path: str, tokenizer: PreTrainedTokenizer): Initializes a SupervisedDataset object.
        __len__(self) -> int: Returns the length of the dataset.
        __getitem__(self, idx) -> Dict[str, torch.Tensor]: Retrieves an example from the dataset at the specified index.
    """

    PROMPT_DICT = {
        'prompt_input':
        ('Below is an instruction that describes a task, paired with an input that provides further context. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
         ),
        'prompt_no_input':
        ('Below is an instruction that describes a task. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Response:'),
    }
    IGNORE_INDEX = -100

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024):
        """Initializes a SupervisedDataset object.

        Args:
            data_path (str): The path to the training data file.
            tokenizer (PreTrainedTokenizer): The tokenizer object used to tokenize the input examples.
        """
        super(SupervisedDataset, self).__init__()
        logger.info(f'Loading dataset from {data_path}')
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            list_data_dict = load_dataset('json',
                                          data_files=data_path)['train']
        else:
            list_data_dict = load_dataset(data_path)['train']

        logger.info('Found %d rows', list_data_dict.num_rows)
        prompt_input, prompt_no_input = self.PROMPT_DICT[
            'prompt_input'], self.PROMPT_DICT['prompt_no_input']
        self.sources = [
            prompt_input.format_map(example) if example.get('input', '') != ''
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        self.targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        self.examples = [s + t for s, t in zip(self.sources, self.targets)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The number of examples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Retrieves an example from the dataset at the specified index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the input_ids, labels, input_len, source_input_ids, and
            source_len tensors.
        """
        example_txt = self.examples[idx]
        # Tokenize the example and source text
        example_tokenized = self.tokenizer(
            example_txt,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
        )
        source_txt = self.sources[idx]
        source_tokenized = self.tokenizer(
            source_txt,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
        )
        # Extract the input_ids tensor
        input_ids = torch.tensor(example_tokenized['input_ids'])
        # Create the labels tensor
        labels = input_ids.clone()
        labels[:len(source_tokenized['input_ids'])] = self.IGNORE_INDEX
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
