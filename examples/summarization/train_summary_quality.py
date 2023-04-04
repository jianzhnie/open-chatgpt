from typing import Any, Dict, Optional, Tuple, Union

import evaluate
import torch
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
import sys

sys.path.append("../../")
from chatgpt.dataset.summarize_dataset import HFSummaryQuality

accuracy = evaluate.load('mse')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return accuracy.compute(predictions=predictions.flatten(),
                            references=labels.flatten())


class QualityTrainer(Trainer):

    def __init__(self, ):
        super().__init__()
        self.loss_fct = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        # forward pass
        outputs = model(**inputs)
        logits = self.sigmoid(outputs.get('logits'))
        loss = self.loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def _compute_loss(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = self.sigmoid(outputs.get('logits'))
        loss = self.loss_fct(logits, labels)

        return loss, logits

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        with torch.no_grad():
            # compute loss on predict data
            loss, logits = self._compute_loss(model, inputs)

        loss = loss.mean().detach()
        labels = inputs['labels']
        if self.args.prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)


if __name__ == '__main__':
    model_name = 'facebook/opt-125m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train = HFSummaryQuality(split='validation',
                             tokenizer=tokenizer,
                             max_length=512)
    eval = HFSummaryQuality(split='test', tokenizer=tokenizer, max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(train.label2idx), problem_type='regression')

    args = TrainingArguments(
        output_dir=f'{model_name}-finetuned',
        num_train_epochs=4,
        warmup_steps=500,
        learning_rate=1e-5,
        fp16=True,
        gradient_checkpointing=False,
        gradient_accumulation_steps=15,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        max_grad_norm=2.0,
        logging_steps=10,
        save_total_limit=4,
        evaluation_strategy='steps',
        eval_steps=200,
        save_steps=1000,
        report_to='wandb',
    )

    trainer = QualityTrainer(
        model,
        args,
        train_dataset=train,
        eval_dataset=eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
