import os
import sys
from transformers import (AutoTokenizer, EarlyStoppingCallback, EvalPrediction,
                          Trainer, TrainingArguments, default_data_collator)

sys.path.append('../../')
from chatgpt.dataset.reward_dataset import PairwiseDataset
from chatgpt.rlhf.reward_model import PairedRewardModel


def compute_metrics(eval_preds: EvalPrediction):
    chosen_reward = eval_preds.predictions[0]  # chosen scores
    rejected_reward = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_reward > rejected_reward) / len(chosen_reward)
    result['accuracy'] = acc

    return result


if __name__ == '__main__':
    optput_dir = 'work_dirs/reward_model_checkpoint'
    if not os.path.exists(optput_dir):
        os.makedirs(optput_dir)

    # Set up the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    # Initialize the reward model from the (supervised) fine-tuned opt model
    reward_model = PairedRewardModel(pretrained='facebook/opt-125m')

    tokenizer.pad_token = tokenizer.eos_token

    # Create the comparisons datasets
    data_path = 'CarperAI/openai_summarize_comparisons'
    # Make pairwise datasets for training
    train_dataset = PairwiseDataset(data_path,
                                    tokenizer,
                                    split='train',
                                    max_length=550)
    val_dataset = PairwiseDataset(data_path,
                                  tokenizer,
                                  split='valid1',
                                  max_length=550)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=optput_dir,
        num_train_epochs=10,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        eval_steps=500,
        save_steps=1000,
        warmup_steps=100,
        learning_rate=1e-5,
        weight_decay=0.0001,  # strength of weight decay
        # half_precision_backend=True,
        # fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        # fp16_opt_level='02',  # mixed precision mode
        do_train=True,  # Perform training
        do_eval=True,  # Perform evaluation
        save_strategy='steps',
        evaluation_strategy='steps',
        save_total_limit=5,
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        logging_steps=50,
        logging_dir='work_dirs/logs',
        # deepspeed="ds_config_rm.json",
        seed=42)

    # Set up the trainer
    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
    trainer.train()
    trainer.save_model(optput_dir)