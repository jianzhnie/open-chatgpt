import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, PreTrainedModel,
                          PreTrainedTokenizer, Trainer, TrainingArguments)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = '</s>'


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={'help': 'Path to the training data.'})


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    model_max_length: int = field(
        default=512,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict,
                                         tokenizer: PreTrainedTokenizer,
                                         model: PreTrainedModel):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

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

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning('Loading data...')
        list_data_dict = load_dataset(data_path)

        logging.warning('Formatting inputs...')
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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        source_txt = self.examples[idx]
        encoding_inputs = self.tokenizer(
            source_txt,
            return_tensors="pt",
            padding='longest',
            max_length=self.tokenizer.model_max_length,
            truncation=True,)

        input_ids = encoding_inputs['input_ids']
        labels = copy.deepcopy(input_ids)
        input_ids_lens = labels_lens = input_ids.ne(self.tokenizer.pad_token_id).sum().item()

        return dict(input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels,
                              batch_first=True,
                              padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side='right',
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in model_args.model_name_or_path:
        tokenizer.add_special_tokens({
            'eos_token': DEFAULT_EOS_TOKEN,
            'bos_token': DEFAULT_BOS_TOKEN,
            'unk_token': DEFAULT_UNK_TOKEN,
        })

    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()


if __name__ == '__main__':
    train()
