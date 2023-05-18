import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser,
                          PreTrainedModel, PreTrainedTokenizer, Trainer,
                          TrainingArguments)

from transformers import LlamaTokenizer



from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

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
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            list_data_dict = load_dataset("json", data_files=data_path)
        else:
            list_data_dict = load_dataset(data_path)['train']

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

        example_txt = self.examples[idx]
        example_tokenized = self.tokenizer(
            example_txt,
            padding='longest',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        input_ids = example_tokenized['input_ids']
        labels = copy.deepcopy(input_ids)
        encoding_input = dict(input_ids=input_ids, labels=labels)
        encoding_input = {
            key: torch.tensor(val)
            for key, val in encoding_input.items()
        }

        return encoding_input


def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        cache_dir=training_args.cache_dir,
    )
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters(
    )  # Be more transparent about the % of trainable params.

    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side='right',
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           pad_to_multiple_of=8,
                                           return_tensors="pt",
                                           padding=True)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict())).__get__(model, type(model))

    trainer.train()
    trainer.save_state()


if __name__ == '__main__':
    train()
