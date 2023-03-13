import os
import sys
import evaluate
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

sys.path.append('../')
from chatgpt.dataset.comparison_dataset import PairwiseDataset
from chatgpt.rlhf.reward_model import RewardModel
from chatgpt.rlhf.pairwise_loss import PairWiseLoss

# Define the metric that we'll use for validation.
accuracy = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    loss_fn = PairWiseLoss()

    def compute_loss(self,
                     model,
                     inputs,
                     loss_fn=loss_fn,
                     return_outputs=False):
        rewards_choosen = model(input_ids=inputs['chosen_input_ids'],
                                attention_mask=inputs['chosen_attention_mask'])
        rewards_rejected = model(
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs['rejected_attention_mask'])

        loss = loss_fn(rewards_choosen, rewards_rejected)
        if return_outputs:
            return loss, {
                'rewards_choosen': rewards_choosen,
                'rewards_rejected': rewards_rejected
            }
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
        fp16=True,
        logging_dir='./logs',
        learning_rate=1e-5,
        save_total_limit=1,
    )

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = RewardModel(model='opt', pretrained='facebook/opt-125m')
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
                                  split='valid1',
                                  max_length=max_length)

    trainer = RewardTrainer(model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            compute_metrics=compute_metrics,
                            eval_dataset=val_dataset,
                            data_collator=default_data_collator)
    trainer.train()
