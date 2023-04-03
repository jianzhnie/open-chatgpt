import os
import random
import sys

import evaluate
import numpy as np
import torch

sys.path.append('../../')
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)

from chatgpt.dataset.summarize_dataset import TLDRDataset


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


if __name__ == '__main__':
    output_dir = 'fintune-summarize-checkpoint'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m',
                                                 use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # Set up the datasets
    data_path = 'CarperAI/openai_summarize_tldr'
    train_dataset = TLDRDataset(data_path, tokenizer, 'train', max_length=512)
    dev_dataset = TLDRDataset(data_path, tokenizer, 'valid', max_length=512)
    # Set up the metric
    rouge = evaluate.load('rouge')

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(predictions,
                                               skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds,
                               references=decoded_labels,
                               use_stemmer=True)
        return result

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_steps=50,
        save_steps=500,
        warmup_steps=100,
        learning_rate=2e-5,
        weight_decay=0.001,
        fp16=True,
        fp16_opt_level='02',  # mixed precision mode
        do_train=True,  # Perform training
        do_eval=True,  # Perform evaluation
        save_strategy='steps',
        save_total_limit=5,
        evaluation_strategy='steps',
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        half_precision_backend=True,
        logging_steps=50,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics)
    trainer.train()
    trainer.save_model(output_dir)
