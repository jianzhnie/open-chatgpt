import os
import sys

from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                          EarlyStoppingCallback, EvalPrediction,
                          default_data_collator)

sys.path.append('../')
from chatgpt.dataset.comparison_dataset import PairwiseDataset
from chatgpt.rlhf.reward_model import RewardModel


def compute_metrics(eval_preds: EvalPrediction):
    chosen_reward = eval_preds.predictions[0]  # chosen scores
    rejected_reward = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_reward > rejected_reward) / len(chosen_reward)
    result["accuracy"] = acc

    return result


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists('rm_checkpoint'):
        os.mkdir('rm_checkpoint')

    training_args = TrainingArguments(
        output_dir='rm_checkpoint/',
        num_train_epochs=10,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        learning_rate=1e-5,
        weight_decay=0.0001,  # strength of weight decay
        fp16=True,
        fp16_opt_level='02',  # mixed precision mode
        do_train=True,  # Perform training
        do_eval=True,  # Perform evaluation
        save_strategy='steps',
        evaluation_strategy='steps',
        save_total_limit=5,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=10,
        seed=42)

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = RewardModel(model='opt', pretrained='facebook/opt-125m')
    # Create the comparisons datasets
    data_path = 'CarperAI/openai_summarize_comparisons'
    # Make pairwise datasets for training
    train_dataset = PairwiseDataset(data_path,
                                    tokenizer,
                                    split='train',
                                    max_length=512)
    val_dataset = PairwiseDataset(data_path,
                                  tokenizer,
                                  split='valid1',
                                  max_length=512)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
    trainer.train()
