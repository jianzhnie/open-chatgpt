import json
from typing import Any, Dict, Union

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def get_dataset_from_jsonl(jsonl_file, return_summary=True):
    # if return_summary is True, return a list of posts with summary concatenated
    # if return_summary is False, return a list of posts and a list of summaries
    with open(jsonl_file, 'r') as f:
        dataset = [json.loads(line) for line in f]
    post_list = []
    summary_list = []
    for d in dataset:
        if return_summary:
            post = f"SUBREDDIT: r/{d['subreddit']}\nTITLE: {d['title']}\nPOST: {d['post']}\nTL;DR: {d['summary']}"
        else:
            post = f"SUBREDDIT: r/{d['subreddit']}\nTITLE: {d['title']}\nPOST: {d['post']}\nTL;DR: "
            summary_list.append(d['summary'])
        post_list.append(post)
    if not return_summary:
        return post_list, summary_list
    return post_list


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


class ComparisonDataset(Dataset):
    """A PyTorch Dataset class for generating input sequences for training a
    text comparison model.

    Args:
        comparison_path (str): Path to the input JSON file.
        tokenizer (tokenizer): A Hugging Face tokenizer object.
        max_length (int, optional): Maximum sequence length for the model.
            Defaults to 550.
    """
    def __init__(self,
                 comparison_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 550):
        # Load the input JSON file into memory
        with open(comparison_path, 'r') as f:
            dataset = [json.loads(line) for line in f]

        # Initialize the class variables
        self.tokenizer = tokenizer
        self.post_list = []
        self.summaries_0 = []
        self.summaries_1 = []
        self.labels = []
        self.max_length = max_length

        # Define a helper function to create the input sequence text
        def make_text(post: Dict[str, Union[str, Dict[str, str]]],
                      summarize: str) -> str:
            return f"SUBREDDIT: r/{post['subreddit']}\nTITLE: {post['title']}\nPOST: {post['post']}\nTL;DR: {summarize}"

        # Parse the JSON input data and generate the input sequences
        for sample in dataset:
            # Add the post to the post list
            self.post_list.append(sample['info']['post'])

            # NOTE: The chosen summary is always the first one, i.e. `sample["summaries"][0]`
            if sample['choice'] == 0:
                # Add the first summary to summaries_0 and the second summary to summaries_1
                self.summaries_0.append(
                    make_text(sample['info'], sample['summaries'][0]['text']))
                self.summaries_1.append(
                    make_text(sample['info'], sample['summaries'][1]['text']))
            else:
                # Add the second summary to summaries_0 and the first summary to summaries_1
                self.summaries_0.append(
                    make_text(sample['info'], sample['summaries'][1]['text']))
                self.summaries_1.append(
                    make_text(sample['info'], sample['summaries'][0]['text']))

            # Add the label (always 0) to the labels list
            self.labels.append(0)

    def __len__(self) -> int:
        return len(self.post_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get the two summaries to compare at the current index
        summ0 = self.summaries_0[idx]
        summ1 = self.summaries_1[idx]
        encodings_dict = self.tokenizer(
            [summ0, summ1],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
        )
        input_ids = torch.tensor(encodings_dict['input_ids'])
        attention_mask = torch.tensor(encodings_dict['attention_mask'])
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class AllSummDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 split: str = 'train',
                 max_length: int = 1024):
        """
        Args:
            data_path (str): Path to the data file.
            tokenizer (Tokenizer): Tokenizer object.
            split (str): Either "train" or "valid".
            max_length (int): Maximum sequence length.
        """
        if split not in ['train', 'valid']:
            raise ValueError(
                "Invalid split. Must be either 'train' or 'valid'.")

        df = pd.read_parquet(data_path)
        if split == 'valid':
            df = df.sample(n=5000)

        self.summaries = [
            f"Summarize: {row['text']}. TL;DR: {row['summary']}"
            for i, row in df.iterrows()
        ]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.summaries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        txt = self.summaries[idx]
        encodings_dict = self.tokenizer(txt,
                                        truncation=True,
                                        max_length=self.max_length,
                                        padding='max_length')

        input_ids = torch.tensor(encodings_dict['input_ids'])
        attn_mask = torch.tensor(encodings_dict['attention_mask'])

        return {'input_ids': input_ids, 'attention_mask': attn_mask}
