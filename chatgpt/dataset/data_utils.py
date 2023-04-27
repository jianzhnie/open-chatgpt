# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Part of the code was adopted from https://github.com/microsoft/Megatron-
DeepSpeed/blob/main/megatron/data/dataset_utils.py."""
import hashlib
import os
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Subset
from transformers import PreTrainedTokenizer

from .raw_datasets import (
    CohereMiracljaqueries2212Dataset, CohereMiraclzhqueries2212Dataset,
    DahoasFullhhrlhfDataset, DahoasRmstaticDataset,
    DahoasSyntheticinstructgptjpairwiseDataset, HelloSimpleAIHC3ChineseDataset,
    LmqgQagjaquadDataset, LmqgQgjaquadDataset, MkqaChineseDataset,
    MkqaJapaneseDataset, OpenaiWebgptcomparisonsDataset, PromptDataset,
    PromptRawDataset, StanfordnlpSHPDataset, Wangrui6ZhihuKOLDataset,
    YitingxieRlhfrewarddatasetsDataset)

name2Method = {
    'Dahoas/rm-static': DahoasRmstaticDataset,
    'Dahoas/full-hh-rlhf': DahoasFullhhrlhfDataset,
    'Dahoas/synthetic-instruct-gptj-pairwise':
    DahoasSyntheticinstructgptjpairwiseDataset,
    'yitingxie/rlhf-reward-datasets': YitingxieRlhfrewarddatasetsDataset,
    'openai/webgpt_comparisons': OpenaiWebgptcomparisonsDataset,
    'stanfordnlp/SHP': StanfordnlpSHPDataset,
    'wangrui6/Zhihu-KOL': Wangrui6ZhihuKOLDataset,
    'Cohere/miracl-zh-queries-22-12': CohereMiraclzhqueries2212Dataset,
    'Hello-SimpleAI/HC3-Chinese': HelloSimpleAIHC3ChineseDataset,
    'mkqa-Chinese': MkqaChineseDataset,
    'mkqa-Japanese': MkqaJapaneseDataset,
    'Cohere/miracl-ja-queries-22-12': CohereMiracljaqueries2212Dataset,
    'lmqg/qg_jaquad': LmqgQgjaquadDataset,
    'lmqg/qag_jaquad': LmqgQagjaquadDataset,
}


def get_raw_dataset(dataset_name, output_path, seed):
    if dataset_name in name2Method:
        return name2Method[dataset_name](output_path, seed, dataset_name)
    else:
        raise RuntimeError(
            f'We do not have configs for dataset {dataset_name}, but you can add it by yourself in py.'
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(output_path, dataset_name, seed, split_name,
                                data_split, split_index, data_size):
    index_file_name = f'{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy'
    if not os.path.isfile(index_file_name):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f'{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy'

            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


def create_dataset_split(
    current_dataset=None,
    raw_dataset: PromptRawDataset = None,
    train_phase: int = 1,
    tokenizer: PreTrainedTokenizer = None,
    max_seq_len: int = 512,
    end_of_conversation_token: str = None,
):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, raw_sample in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(raw_sample)
            # the accept response
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_dataset.append(chosen_sentence)

    elif train_phase == 2:
        for i, raw_sample in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(raw_sample)
            # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(raw_sample)
            # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_dataset.append(chosen_sentence)
                reject_dataset.append(reject_sentence)

    elif train_phase == 3:
        for i, raw_sample in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(raw_sample)
            if prompt is not None:
                prompt_dataset.append(prompt)
    return PromptDataset(
        prompt_dataset,
        chosen_dataset,
        reject_dataset,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        train_phase=train_phase,
    )


def create_dataset(dataset_name: str = None,
                   data_split: str = '10,0',
                   output_path: str = None,
                   train_phase: int = None,
                   seed: int = None,
                   tokenizer: PreTrainedTokenizer = None,
                   max_seq_len: int = 512,
                   end_of_conversation_token: str = None):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed)
    print(raw_dataset)
    assert isinstance(raw_dataset, PromptRawDataset)
    # assert raw_dataset in name2Method.values()
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, 'train', data_split,
                                              train_phase - 1,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(
        train_dataset,
        raw_dataset,
        train_phase,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        end_of_conversation_token=end_of_conversation_token)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, 'eval',
                                             data_split, train_phase - 1,
                                             len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(
        eval_dataset,
        raw_dataset,
        train_phase,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        end_of_conversation_token=end_of_conversation_token)
    print(train_dataset)
    return train_dataset, eval_dataset


def create_prompt_dataset(
    dataset_names: str = None,
    data_split: str = None,
    output_path: str = None,
    train_phase: int = None,
    seed: int = None,
    tokenizer: PreTrainedTokenizer = None,
    max_seq_len: int = 512,
    end_of_conversation_token='<|endoftext|>',
    sft_only_data_path=[],
):
    """Creates the prompt dataset."""
    os.makedirs(output_path, exist_ok=True)
    fname = '_'.join(dataset_names)
    sft_cache_key = '_'.join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs['name_or_path'].replace('/', '_')
    fname = f'{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}'

    fname = '_'.join(fname.split('/'))
    fname = hashlib.sha256(fname.encode()).hexdigest()
    # hash the file name to avoid too long file name
    train_fname = f'{output_path}/traindata_{fname}.pt'
    eval_fname = f'{output_path}/evaldata_{fname}.pt'

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)

    # Skip creating cache if we found it on all the nodes.
    if cache_found:
        return torch.load(train_fname), torch.load(eval_fname)
    else:
        if len(dataset_names) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                dataset_names[0], data_split, output_path, train_phase, seed,
                end_of_conversation_token)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_name in dataset_names:
                train_dataset, eval_dataset = create_dataset(
                    d_name,
                    data_split,
                    output_path,
                    train_phase,
                    seed,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    end_of_conversation_token=end_of_conversation_token)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        if train_phase == 1 and sft_only_data_path:
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    sft_path,
                    '10,0,0',
                    output_path,
                    train_phase,
                    seed,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    end_of_conversation_token=end_of_conversation_token,
                )
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    return torch.load(train_fname), torch.load(eval_fname)


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch['input_ids'] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch['attention_mask'] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        # make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch['prompt'] = F.pad(prompt,
                                    pad=(pad_length, 0),
                                    mode='constant',
                                    value=pad_token_id)
            batch['prompt_att_mask'] = F.pad(prompt_mask,
                                             pad=(pad_length, 0),
                                             mode='constant',
                                             value=0)
        else:
            batch['prompt'] = prompt
            batch['prompt_att_mask'] = prompt_mask
        batch['prompt'] = batch['prompt'].flip(1)
        batch['prompt_att_mask'] = batch['prompt_att_mask'].flip(1)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets['train'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc='Running tokenizer on dataset',
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f'Grouping texts in chunks of {block_size}',
    )

    train_dataset = lm_datasets['train']

    return train_dataset

