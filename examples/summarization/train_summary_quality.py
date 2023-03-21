import evaluate
import torch
from torch import nn
from transformers import Trainer

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
