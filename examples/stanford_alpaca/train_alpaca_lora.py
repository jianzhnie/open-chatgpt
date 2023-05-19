import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LlamaForCausalLM, HfArgumentParser, LlamaTokenizer,
                          PreTrainedTokenizer, Trainer, TrainingArguments)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'


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


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj'])
    lora_weight_path: str = ''
    bias: str = 'none'


def maybe_zero_3(param):
    if hasattr(param, 'ds_id'):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.cpu().clone().detach()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(state_dict, bias):
    if bias == 'none':
        to_return = {
            k: state_dict[k].cpu().clone().detach()
            for k in state_dict if 'lora_' in k
        }
    elif bias == 'all':
        to_return = {
            k: state_dict[k]
            for k in state_dict if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        for k in state_dict:
            if 'lora_' in k:
                to_return[k] = state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


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

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_seq_length: int = 512):
        super(SupervisedDataset, self).__init__()
        logging.warning('Loading data...')
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            list_data_dict = load_dataset('json',
                                          data_files=data_path)['train']
        else:
            list_data_dict = load_dataset(data_path)['train']

        logging.warning('Formatting inputs...')
        prompt_input, prompt_no_input = self.PROMPT_DICT[
            'prompt_input'], self.PROMPT_DICT['prompt_no_input']

        self.prompts = [
            prompt_input.format_map(example) if example.get('input', '') != ''
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        self.responses = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        self.prompt_and_responses = [
            s + t for s, t in zip(self.prompts, self.responses)
        ]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.prompt_and_responses)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        prompt_and_response = self.prompt_and_responses[idx]
        prompt = self.prompts[idx]

        tokenized_prompt_and_response = self.tokenizer(
            prompt_and_response,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
        )
        tokenized_prompt = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
        )

        # The labels are the full prompt with response, but with the prompt masked out
        input_ids = torch.tensor(tokenized_prompt_and_response['input_ids'])
        attn_masks = torch.tensor(
            tokenized_prompt_and_response['attention_mask'])
        labels = input_ids.clone()
        labels[:len(tokenized_prompt['input_ids'])] = self.IGNORE_INDEX
        return {
            'input_ids': input_ids,
            'attention_mask': attn_masks,
            'labels': labels,
        }


def train(model_args: ModelArguments, data_args: DataArguments,
          training_args: TrainingArguments, lora_args: LoraArguments) -> None:
    """
    Trains a language model using Hugging Face's Transformers library.

    Args:
        model_args (ModelArguments): The arguments for the model configuration.
        data_args (DataArguments): The arguments for the data configuration.
        training_args (TrainingArguments): The arguments for the training configuration.

    Returns:
        None

    """
    device_map = 'auto'
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # Load the pre-trained model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
    )

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type='CAUSAL_LM',
    )

    if training_args.gradient_checkpointing:
        logging.warning(
            'gradient checkpointing with lora makes requires_grad '
            'incorrect and needs a monkey patch in Trainer or the '
            "wrapped model's forward. ref: "
            'https://github.com/lm-sys/FastChat/pull/138#issuecomment-1509172198'
        )

    if model.config.model_type == 'llama':
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side='right',
            use_fast=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side='right',
            use_fast=True,
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

    if len(special_tokens_dict) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    # Be more transparent about the % of trainable params.
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    train_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    if training_args.resume_from_checkpoint and list(
            pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    # Save the trained model
    trainer.save_model(training_args.output_dir)

    # Save states. Weights might be a placeholder in zero3 and need a gather
    state_dict = get_peft_state_maybe_zero_3(model.state_dict(),
                                             lora_args.bias)
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses(
    )

    train(model_args, data_args, training_args, lora_args)
