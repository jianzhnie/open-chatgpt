import os
import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments
import evaluate
import numpy as np
from chatgpt.dataset.comparison_dataset import PairwiseDataset
from chatgpt.rlhf.reward_model import GPTRewardModel

# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"],
                          attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"],
                          attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists('rm_checkpoint'):
        os.mkdir('rm_checkpoint')

    training_args = TrainingArguments(
        output_dir='rm_checkpoint/',
        num_train_epochs=5,
        logging_steps=10,
        gradient_accumulation_steps=2,
        save_strategy='steps',
        evaluation_strategy='steps',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir='./logs',
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed='ds_config_gpt_j.json',
        save_total_limit=1,
    )

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = GPTRewardModel(model_path='facebook/opt-125m')

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Create the comparisons datasets
    data_path = 'CarperAI/openai_summarize_comparisons'
    # Make pairwise datasets for training
    max_length = 550
    train_dataset = PairwiseDataset(data_path,
                                    tokenizer,
                                    split='train',
                                    max_length=max_length)
    val_dataset = PairwiseDataset(data_path,
                                  tokenizer,
                                  split='valid',
                                  max_length=max_length)

    trainer = RewardTrainer(model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            compute_metrics=compute_metrics,
                            eval_dataset=val_dataset,
                            daata_collator=None)
    trainer.train()
