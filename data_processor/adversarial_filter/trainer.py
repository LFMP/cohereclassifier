from torch.nn import CrossEntropyLoss
from transformers import Trainer


class WeightedTrainer(Trainer):
  # replace the loss function to weighted CrossEntropyLoss for classification with
  # imbalanced classes
  def __init__(self, *args, class_weights, **kwargs):
    super().__init__(*args, **kwargs)
    if class_weights is not None:
      self.class_weights = class_weights

  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    input = logits.view(-1, 2).to(model.device)
    target = labels.view(-1, 2).to(model.device)
    loss_fn = CrossEntropyLoss(weight=self.class_weights.to(model.device))
    loss = loss_fn(input, target)
    return (loss, outputs) if return_outputs else loss
