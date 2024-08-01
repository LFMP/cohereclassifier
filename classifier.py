import argparse
import os
import time
from typing import Optional

import datasets
import numpy as np
import torch
from datasets import ClassLabel
from loguru import logger
from peft import LoKrConfig, LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from torch.nn import CrossEntropyLoss
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, SchedulerType, Trainer,
                          TrainerCallback, TrainingArguments)
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import PredictionOutput

import wandb
from data_processor.ftbr import FakeTrueBr
from data_processor.gcdc import GCDC
from data_processor.pos_tags import POS_TAGS_COMPILED
from data_processor.rst_tags import RST_TAGS_COMPILED

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_path',
    type=str,
    required=True,
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=5,
)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--tokenizer',
                    type=str,
                    default="severinsimmler/xlm-roberta-longformer-base-16384")
parser.add_argument('--rst', action='store_true')
parser.add_argument('--pos', action='store_true')
parser.add_argument('--lokr', action='store_true')
parser.add_argument('--lora', action='store_true')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--num_cycles', type=int, default=2)
parser.add_argument('--processed', action='store_true')
parser.add_argument('--runs', type=int, default=1)
args = parser.parse_args()


def load_dataset(dataset_path=args.dataset_path):
  # get the name of the dataset which is the last part of the normalized path
  normalized_path = os.path.normpath(dataset_path)
  dataset_name = os.path.basename(normalized_path).lower()
  if args.processed:
    dataset = datasets.load_from_disk(dataset_path)
  else:
    if "gcdc" in dataset_name:
      # load dataset to process
      dataset = GCDC(dataset_path, batch_size=args.batch_size).load_dataset()
      logger.info(
          "Saving processed dataset to disk for future use in data/gcdc")
      dataset.save_to_disk("data/gcdc")
    elif "faketrue" in dataset_name:
      dataset = FakeTrueBr(dataset_path,
                           batch_size=args.batch_size).load_dataset()
      logger.info(
          "Saving processed dataset to disk for future use in data/faketrue")
      dataset.save_to_disk("data/faketrue")
  return dataset


def tokenize_function(examples, field="text"):
  return tokenizer(examples[field])


def collate_fn(examples):
  texts = torch.tensor([example["text"] for example in examples])
  labels = torch.tensor([example["label"] for example in examples])
  return {"texts": texts, "labels": labels}


def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  metrics = {}
  metrics["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
  metrics["f1"] = f1_score(labels,
                           predictions,
                           average="weighted",
                           zero_division=0.0)
  metrics["accuracy"] = accuracy_score(labels, predictions)
  return metrics


def add_tokens():
  embedding_size = len(tokenizer)
  num_new_tokens = 0
  if args.rst:
    logger.info("Adding RST tags to tokenizer")
    num_new_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": RST_TAGS_COMPILED},
        replace_additional_special_tokens=False,
    )
    logger.success(
        f"{num_new_tokens} RST tags added to tokenizer (from {embedding_size}) to {len(tokenizer)})"
    )
  elif args.pos:
    logger.info("Adding POS tags to tokenizer")
    num_new_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": POS_TAGS_COMPILED},
        replace_additional_special_tokens=False,
    )
    logger.success(
        f"{num_new_tokens} POS tags added to tokenizer (from {embedding_size}) to {len(tokenizer)})"
    )

  logger.info("Resizing token embeddings of model")
  model.resize_token_embeddings(len(tokenizer))
  logger.success(
      f"Model token embeddings resized from {embedding_size} to {len(tokenizer)}"
  )
  logger.info("Overwriting the embeddings to have better results")
  input_embeddings = model.get_input_embeddings().weight.data
  input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0,
                                                                 keepdim=True)
  input_embeddings[-num_new_tokens:] = input_embeddings_avg
  logger.success("Embeddings overwritten")


class LoggingCalback(TrainerCallback):
  # create a callback callend in end of each epoch to log confusion matrix to
  # wandb
  def __init__(self, trainer):
    self._trainer = trainer

  def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, **kwargs):
    # predict on training dataset and log metrics to wandb
    pred = self._trainer.predict(
        test_dataset=self._trainer.train_dataset,
        metric_key_prefix="train",
    )
    # metrics = pred.metrics
    # replace the "_" in keys with "/" to avoid wandb error
    # metrics = {key.replace("_", "/", 1): metrics[key] for key in metrics}
    # wandb.log(metrics)
    # log accuracy by class to wandb
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = pred.label_ids
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc = cm.diagonal()
    data = [[idx, acc[idx]] for idx in range(len(acc))]
    table = wandb.Table(data=data, columns=["label", "value"])
    wandb.log({
        "train/accuracy_by_class":
            wandb.plot.bar(
                table,
                "label",
                "value",
                title="Accuracy by class",
            ),
    })
    wandb.log({
        "train/confusion_matrix":
            wandb.plot.confusion_matrix(
                preds=y_pred,
                y_true=y_true,
            ),
    })

  def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                  control: TrainerControl, **kwargs) -> None:
    # call evaluation and get y_true and y_pred for evaluation dataset
    pred = self._trainer.predict(
        test_dataset=self._trainer.eval_dataset,
        metric_key_prefix="eval",
    )
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = pred.label_ids
    # log accuracy by class to wandb
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc = cm.diagonal()
    data = [[idx, acc[idx]] for idx in range(len(acc))]
    table = wandb.Table(data=data, columns=["label", "value"])
    wandb.log({
        "eval/accuracy_by_class":
            wandb.plot.bar(
                table,
                "label",
                "value",
                title="Accuracy by class",
            ),
    })
    wandb.log({
        "eval/confusion_matrix":
            wandb.plot.confusion_matrix(
                preds=y_pred,
                y_true=y_true,
            ),
    })


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
    input = logits.view(-1, self.model.config.num_labels)
    target = labels.view(-1).to(model.device)
    loss_fn = CrossEntropyLoss(weight=self.class_weights)
    loss = loss_fn(input, target)
    return (loss, outputs) if return_outputs else loss


normalized_path = os.path.normpath(args.dataset_path)
dataset_name = os.path.basename(normalized_path).lower()

logger.info(f"Loading dataset {dataset_name}")
dataset = load_dataset()
for d in dataset:
  # actual gcdc labels are {0,1,2}, transformed to [0, 2]
  dataset[d] = dataset[d].filter(lambda e: e["label"] != 1)
  # tranform labels to [0, 1]
  dataset[d] = dataset[d].map(
      lambda e: {"label": 0 if e["label"] == 0 else 1},
      remove_columns=["label"],
      num_proc=4,
      desc="Transforming labels to [0, 1]",
  )
  dataset[d] = dataset[d].cast_column("label", ClassLabel(names=["0", "1"]))
num_labels = dataset["train"].features["label"].num_classes
logger.info(f"Number of labels: {num_labels}")
logger.success("Dataset loaded")

logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
logger.info("Tokenizer loaded")

logger.info("Loading model")
model = AutoModelForSequenceClassification.from_pretrained(
    "severinsimmler/xlm-roberta-longformer-base-16384",
    num_labels=num_labels,
)
logger.success("Model loaded")

# calculate class weights for imbalanced datasets
class_weights = None
if "train" in dataset:
  labels = dataset["train"]["label"]
  class_weights = torch.tensor([1 / count for count in np.bincount(labels)],
                               device="cuda")
  class_weights = class_weights / class_weights.sum()
  # convert tensor to float
  class_weights = class_weights.float()
  logger.info(f"Class weights: {class_weights}")

target_modules: list[str] = [
    "query", "key", "value", "query_global", "key_global", "value_global"
]

if args.rst or args.pos:
  add_tokens()
  target_modules += ["embed_tokens", "lm_head"]

if "text" in dataset["train"].column_names:
  text_field = "text"
elif "story" in dataset["train"].column_names:
  text_field = "story"
else:
  raise ValueError(
      "Dataset does not have a default column field ('text' or 'story')")

tags = [dataset_name]
logger.info("Tokenizing dataset")
if args.rst:
  tokenized_field = f"{text_field}_rst_mixed"
  tags.append("rst")
elif args.pos:
  tokenized_field = f"{text_field}_pos_mixed"
  tags.append("pos")
else:
  tokenized_field = text_field
  tags.append("vanilla")
tokenized_datasets = dataset.map(tokenize_function,
                                 batch_size=args.batch_size,
                                 batched=True,
                                 fn_kwargs={"field": tokenized_field})
tokenized_datasets = tokenized_datasets.remove_columns([text_field])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
logger.success("Dataset tokenized")

if args.lokr:
  tags.append("lokr")
  logger.info("Creating Lokr model")
  config = LoKrConfig(
      r=16,
      alpha=16,
      target_modules=target_modules,
      inference_mode=False,
      module_dropout=0.1,
      task_type=TaskType.SEQ_CLS,
      modules_to_save=["classifier"],
  )
  model = get_peft_model(model, config)
  logger.success("Lokr model created")
  model.print_trainable_parameters()
elif args.lora:
  tags.append("lora")
  logger.info("Creating Lora model")
  config = LoraConfig(
      r=16,
      lora_alpha=16,
      target_modules=target_modules,
      inference_mode=False,
      lora_dropout=0.1,
      bias="none",
      task_type=TaskType.SEQ_CLS,
      modules_to_save=["classifier"],
  )
  model = get_peft_model(model, config)
  logger.success("Lora model created")
  model.print_trainable_parameters()

for run_idx in range(args.runs):
  # define run name
  run_init_time = time.strftime("%Y-%m-%d_%H-%M-%S")

  group_name = (f"{dataset_name}_"
                f"epochs-{args.epochs}_"
                f"lokr-{args.lokr}_"
                f"lora-{args.lora}_"
                f"batch_size-{args.batch_size}_"
                f"num_cycles-{args.num_cycles}_"
                f"rst-{args.rst}_"
                f"pos-{args.pos}")

  prefix_run = f"{group_name}_run-{run_idx}"

  out_dir = f"checkpoints/{prefix_run}"
  run_name = f"{prefix_run}_{run_init_time}"
  logger.info(f"Run name: {run_name}")

  data_collator = DataCollatorWithPadding(tokenizer)

  # initialize wandb with project name, tags and group
  wandb.init(project=f"binary-coherence-classification-{dataset_name}",
             reinit=True,
             group=group_name,
             tags=tags)

  # Define training arguments
  training_args = TrainingArguments(
      output_dir=out_dir,
      logging_dir="./logs",
      eval_strategy="epoch",
      save_strategy="epoch",
      logging_steps=1,
      do_eval=True,
      load_best_model_at_end=True,
      overwrite_output_dir=True,
      learning_rate=args.lr,
      per_device_train_batch_size=args.batch_size,
      per_device_eval_batch_size=args.batch_size,
      bf16=True,
      bf16_full_eval=True,
      num_train_epochs=args.epochs,
      warmup_steps=50,
      lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
      lr_scheduler_kwargs={"num_cycles": args.num_cycles},
      gradient_checkpointing=True,
      gradient_checkpointing_kwargs={"use_reentrant": False},
      label_names=["labels"],
      report_to="wandb",
      metric_for_best_model="eval_balanced_accuracy",
      run_name=run_name,
  )

  trainer = WeightedTrainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["validation"],
      data_collator=data_collator,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
      class_weights=class_weights,
  )

  trainer.add_callback(LoggingCalback(trainer))

  logger.info("Training model")
  trainer.train()
  logger.success("Model trained")

  logger.info("Evaluating model")
  eval_results = trainer.evaluate()
  logger.success(f"Eval results: {eval_results}")

  logger.info("Predicting on evaluation dataset by source")
  # get the unique sources in the evaluation dataset
  sources = tokenized_datasets["validation"].to_pandas()["source_name"].unique()
  sources = sources.tolist()
  dataset_by_source = {}
  for source in sources:
    dataset_by_source[source] = tokenized_datasets["validation"].filter(
        lambda x: x["source_name"] == source)

  # predict on each source and log metrics to wandb
  for source in sources:
    pred: PredictionOutput = trainer.predict(
        test_dataset=dataset_by_source[source],
        metric_key_prefix=f"eval/{source}",
    )
    metrics = pred.metrics
    # create table for metrics to log to wandb
    data = [[key, metrics[key]] for key in metrics]
    columns = ["metric", "value"]
    table = wandb.Table(data=data, columns=columns)
    wandb.log({f"eval/{source}/metrics": table})
    logger.success(f"Metrics for {source}: {metrics}")

  logger.info("Saving model")
  trainer.model.save_pretrained(out_dir)
  logger.info("Model saved")

  if args.lora or args.lokr:
    logger.info("Loading base model and merging with PEFT model")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "severinsimmler/xlm-roberta-longformer-base-16384",
        num_labels=num_labels,
    )
    merged_model = PeftModel.from_pretrained(base_model, out_dir)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(out_dir + "_merged")
    logger.success("Model merged and saved")
