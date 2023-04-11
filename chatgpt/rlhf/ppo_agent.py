import inspect
import os
import time
import warnings
from typing import Callable, List, Optional, Union

import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (DataCollatorForLanguageModeling, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from chatgpt.utils.modeling import AdaptiveKLController, FixedKLController


class PPOTrainer():
    def __init__(
            self,
            config,
            model: nn.Module = None,
            ref_model: nn.Module = None,
            tokenizer: Union[PreTrainedTokenizer,
                             PreTrainedTokenizerFast] = None,
            dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
            optimizer: Optional[optim.Optimizer] = None,
            data_collator=None,
            lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
            device: str = None):
        super().__init__()

        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            **config.accelerator_kwargs,
        )

        self.accelerator.init_trackers(config.tracker_project_name,
                                       config=config.to_dict(),
                                       init_kwargs=config.tracker_kwargs)
        self.model = model
        self.ref_model = ref_model

        if not (isinstance(tokenizer, PreTrainedTokenizer)
                or isinstance(tokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                'tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast'
            )
        self.tokenizer = tokenizer
        if dataset is not None and not (isinstance(dataset,
                                                   torch.utils.data.Dataset)
                                        or isinstance(dataset, Dataset)):
            raise ValueError(
                'dataloader must be a torch.utils.data.Dataset or datasets.Dataset'
            )
        elif dataset is None:
            warnings.warn(
                'No dataset is provided. Make sure to set config.batch_size to the correct value before training.',
                UserWarning,
            )
        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset,
                                                      data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            warnings.warn(
                'No dataset is provided. In a multi-GPU setting, this will lead to an error. You should',
                ' prepare your dataloader yourself with `dataloader = ppo_trainer.accelerator.prepare(dataloader)`',
                ' and using `torch.utils.data.DataLoader`, or pass a dataset to the `PPOTrainer`. Please ',
                ' refer to the documentation for more details.',
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        # Step 3: Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer,
                                                             mlm=False)
        if optimizer is None:
            self.optimizer = Adam(filter(lambda p: p.requires_grad,
                                         self.model.parameters()),
                                  lr=self.config.learning_rate)
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = optim.lr_scheduler._LRScheduler

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    'lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)'
                )
        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef,
                                               self.config.target,
                                               self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        # Safety checkers for DS integration
        is_deepspeed_used = self.accelerator.distributed_type == 'DEEPSPEED' and hasattr(
            self.accelerator.state, 'deepspeed_plugin')

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(self.model, self.optimizer,
                                     self.data_collator, self.dataloader,
                                     self.lr_scheduler)
        if is_deepspeed_used:
            # 8 bit models are already set on the correct device
            if not getattr(self.ref_model.pretrained_model,
                           'is_loaded_in_8bit', False):
                # DS integration only allows for single model and as `ref_model` is only used for
                # `KL devergence loss`,i.e, in eval model, just have it be on the respective device and
                # there is no need to pass it to the `accelerator.prepare` call
                self.ref_model = self.ref_model.to(self.accelerator.device)

            # this hack seems to be needed for DS stage 3 to work
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3:
                self.model.train()
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

    def prepare_dataloader(self,
                           dataset: Union[torch.utils.data.Dataset, Dataset],
                           data_collator=None):
        """Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """Generate response with the model given the query tensor. call the
        `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`batch_size`, `seq_len`) containing query tokens.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """

        if isinstance(query_tensor, List):
            return self._generate_batched(
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )

        else:
            if length_sampler is not None:
                generation_kwargs['max_new_tokens'] = length_sampler()
            response = self.accelerator.unwrap_model(self.model).generate(
                input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs)

            if not return_prompt and not self.is_encoder_decoder:
                return response[:, query_tensor.shape[0]:]
            return response

    def _generate_batched(
        self,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        **generation_kwargs,
    ):
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs['max_new_tokens'] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {'input_ids': batch, 'attention_mask': batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors='pt',
            ).to(self.current_device)

            generations = self.accelerator.unwrap_model(
                self.model).generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations,
                                        padded_inputs['attention_mask']):
                output = generation[(1 - mask).sum():]  # remove padding
                if not return_prompt and not self.is_encoder_decoder:
                    output = output[(mask).sum():]  # remove prompt
                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    def step_batch(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        """Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
        Returns:
            `tuple`: The input processed data.
        """
        for name, tensor_list in zip(['queries', 'responses', 'scores'],
                                     [queries, responses, scores]):
            if not isinstance(tensor_list, list):
                raise ValueError(
                    f'{name} must be a list of tensors - got {type(tensor_list)}'
                )
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(
                    f'Elements in {name} must tensors - got {type(tensor_list[0])}'
                )
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f'Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}'
                )

        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(
                    f'Scores must be 1-dimensional - got {score.dim()} for {score}'
                )
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores

    def step(self, queries: List[torch.LongTensor],
             responses: List[torch.LongTensor],
             scores: List[torch.FloatTensor]):
        """Train the model on a batch of queries, responses and scores.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
        Returns:
            `tuple`: A tuple containing the loss and the number of examples.
        """
        queries, responses, scores = self.step_batch(self.config.batch_size,
                                                     queries, responses,
                                                     scores)

        model_inputs = self.prepare_model_inputs(queries, responses)

        # forward pass
        loss = self.accelerator.unwrap_model(self.model)(
            input_ids=queries,
            labels=responses,
            scores=scores,
            return_dict=True,
        ).loss

        # backward pass
        self.accelerator.backward(loss)

        return loss, len(queries)
