import argparse
from itertools import combinations
from random import shuffle
from typing import Any, Dict, List

import datasets
import evaluate
import numpy as np
import torch
from data_collator import DataCollatorForMultipleChoice
from datasets import Dataset, DatasetDict
from loguru import logger
from peft import LoKrConfig, PeftModel, TaskType, get_peft_model
from processor import batched_encode_function, batched_preprocess_function
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from tqdm import tqdm
from trainer import WeightedTrainer
from transformers import (AutoModelForMultipleChoice, AutoTokenizer,
                          TrainingArguments)
from transformers.trainer_utils import EvalPrediction

import wandb


def argument_parser():
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
  parser.add_argument(
      '--tokenizer',
      type=str,
      default="severinsimmler/xlm-roberta-longformer-base-16384")
  parser.add_argument('--max_seq_length', type=int, default=16384)
  parser.add_argument('--generated_texts', type=int, default=12)
  parser.add_argument('--separator', type=str, default='||')
  parser.add_argument('--iterations', type=int, default=40)
  parser.add_argument('--num_texts', type=int, default=3)
  parser.add_argument('--num_combinations', type=int, default=10)
  parser.add_argument('--lokr', action='store_true')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  return args


def load_dataset():
  logger.info("Loading dataset")
  dataset = datasets.load_from_disk(args.dataset_path)
  logger.success("Dataset loaded")
  return dataset


def batch_inference_function(examples: Dict[str, Any]) -> Dict[str, Any]:
  # generate the combinations of true_texts with other texts
  batch_size: int = len(examples[list(examples.keys())[0]])
  texts_columns: List[str] = [clm for clm in examples.keys() if 'text_' in clm]
  texts_to_choose: List[int] = range(len(texts_columns))
  for idx_e in range(batch_size):
    true_texts = examples['ids_true'][idx_e]
    if type(true_texts) == torch.Tensor:
      false_texts: List[int] = list(
          set(texts_to_choose) - set(true_texts.detach().tolist()))
    else:
      false_texts: List[int] = list(set(texts_to_choose) - set(true_texts))
    all_combs: List[List[int]] = []
    for idx_t in true_texts:
      combs: List[List[int]] = combinations(false_texts, args.num_texts)
      combs = [
          np.random.permutation(list(comb) + [idx_t.item()]) for comb in combs
      ]
      all_combs.extend(combs)
    if args.num_combinations:
      # suffle the combinations
      shuffle(all_combs)
      all_combs = all_combs[:args.num_combinations]

    for comb in all_combs:
      # get tensors using comb as index
      comb = torch.tensor(comb, dtype=torch.int)
      model_input = {
          'input_ids':
              torch.index_select(examples['input_ids'][idx_e], 0, comb),
          'attention_mask':
              torch.index_select(examples['attention_mask'][idx_e], 0, comb),
          'labels':
              torch.index_select(examples['labels'][idx_e], 0, comb)
      }
      with torch.no_grad():
        model_output = model(**{
            k: v.unsqueeze(0).to("cuda") for k, v in model_input.items()
        })
      # get the logits
      logits = model_output.logits
      true_skipped = False
      for order, id in enumerate(comb):
        if id not in true_texts:
          actual_logit = examples[f'best_logit_{id}'][idx_e]
          if type(actual_logit) != torch.Tensor:
            actual_logit = torch.tensor(actual_logit)
          new_logit = logits[0][order]
          if not actual_logit.is_nonzero() or actual_logit < new_logit:
            examples[f'best_logit_{id}'][idx_e] = new_logit
            if -1 in examples[f'best_texts'][idx_e]:
              comb_texts = set(comb.clone().detach().tolist()) - set(
                  true_texts.detach().tolist())
              initial_comb = torch.tensor(list(comb_texts), dtype=torch.int)
              examples[f'best_texts'][idx_e] = initial_comb
              break
            elif id not in examples[f'best_texts'][idx_e]:
              if true_skipped:
                examples[f'best_texts'][idx_e][order - 1] = id
              else:
                examples[f'best_texts'][idx_e][order] = id
        else:
          true_skipped = True
  return examples


def update_dataset(dataset: DatasetDict) -> Dataset:
  # Update the dataset with the new logits to be used in the next iteration
  ds_val: Dataset = dataset['val']
  ds_test: Dataset = dataset['test']
  logger.info("Updating validation")
  ds_val = ds_val.map(
      batch_inference_function,
      batched=True,
      batch_size=args.batch_size,
      load_from_cache_file=False,
      num_proc=1,
      desc="Updating validation",
  )
  dataset['val'] = ds_val
  logger.info("Updating test")
  ds_test = ds_test.map(
      batch_inference_function,
      batched=True,
      batch_size=args.batch_size,
      num_proc=1,
      load_from_cache_file=False,
      desc="Updating test",
  )
  dataset['test'] = ds_test
  return datasets.concatenate_datasets([
      dataset['train'],
      dataset['val'],
      dataset['test'],
  ])


def compute_meteor(dataset1, dataset2):
  best_on_left = []
  best_on_right = []
  for row in tqdm(range(len(dataset1)), desc="Computing meteor"):
    best_texts_left: List[int] = dataset1[row]['best_texts']
    best_texts_right: List[int] = dataset2[row]['best_texts']
    if -1 not in best_texts_left:
      true_text_left = dataset1[row]['labels'].argwhere().flatten()
      true_text_left = dataset1[row][f'text_{true_text_left[0]}'] if dataset1[
          row][f'is_artificial_{true_text_left[0]}'] else dataset1[row][
              f'text_{true_text_left[1]}']
      texts_left = [dataset1[row][f"text_{id}"] for id in best_texts_left]
      reference = [true_text_left for _ in range(len(texts_left))]
      meteor_left = meteor.compute(
          predictions=texts_left,
          references=reference,
      )
      best_on_left.append(meteor_left['meteor'])
    if -1 not in best_texts_right:
      true_text_right = dataset2[row]['labels'].argwhere().flatten()
      true_text_right = dataset2[row][f'text_{true_text_right[0]}'] if dataset2[
          row][f'is_artificial_{true_text_right[0]}'] else dataset2[row][
              f'text_{true_text_right[1]}']
      texts_right = [dataset2[row][f"text_{id}"] for id in best_texts_right]
      reference = [true_text_right for _ in range(len(texts_right))]
      meteor_right = meteor.compute(
          predictions=texts_right,
          references=reference,
      )
      best_on_right.append(meteor_right['meteor'])
  mean_left = np.mean(best_on_left) if best_on_left != [] else 0
  mean_right = np.mean(best_on_right) if best_on_right != [] else 0
  return (mean_left, mean_right)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, Any]:
  predictions = eval_pred.predictions
  labels = eval_pred.label_ids
  inputs = eval_pred.input
  max_logits = np.array(predictions).max(axis=1).reshape(-1, 1)
  predictions = np.where(predictions == max_logits, 1, 0).flatten()
  labels = np.array(labels).flatten()
  matrix = confusion_matrix(labels, predictions)
  acc_by_class = np.divide(matrix.diagonal(), matrix.sum(axis=1))
  metrics = {}
  metrics["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
  metrics["f1"] = f1_score(labels,
                           predictions,
                           average="weighted",
                           zero_division=0.0)
  metrics["accuracy"] = accuracy_score(labels, predictions)
  for idx, acc in enumerate(acc_by_class):
    metrics[f"accuracy_class_{idx}"] = acc
  # compute meteor
  true_texts = inputs['labels'].argwhere().flatten()
  reference = [inputs[i] for i in true_texts] * len(inputs)
  preds = [inputs[i] for i in range(len(inputs)) if i not in true_texts]
  meteor_score = meteor.compute(predictions=preds, references=reference)
  metrics["meteor"] = meteor_score['meteor']
  return metrics


def best_dataset(dataset1: Dataset,
                 dataset2: Dataset,
                 criteria: str = "mean") -> Dataset:

  # return True if logits of texts in dataset2['best_texts'] are greater
  # than the logits of texts in dataset1['best_texts']
  meteor_left, meteor_right = compute_meteor(dataset1, dataset2)
  wandb.log({"previous_meteor_score": meteor_right})
  wandb.log({"new_meteor_score": meteor_right})
  logger.success(f"Mean meteor of left: {meteor_left}")
  logger.success(f"Mean meteor of right: {meteor_right}")
  if criteria.lower() == "sum":
    best_on_left: int = 0
    best_on_right: int = 0
    logger.info("Comparing datasets using sum criteria")
    for row in tqdm(range(len(dataset1)), desc="Iterating over rows"):
      best_texts_left = dataset1[row]['best_texts']
      best_texts_right = dataset2[row]['best_texts']
      for idx in range(len(best_texts_left)):
        if best_texts_left[idx] != -1 and best_texts_right[idx] != -1:
          if dataset1[row][f'best_logit_{best_texts_left[idx]}'] > dataset2[
              row][f'best_logit_{best_texts_right[idx]}']:
            best_on_left += 1
          else:
            best_on_right += 1
    wandb.log({"sum_logits": best_on_right})
    logger.success(f"Sum of previous: {np.mean(best_on_left)}")
    logger.success(f"Sum of actual: {np.mean(best_on_right)}")
    if best_on_left > best_on_right:
      logger.success("Previous dataset is better")
      return dataset1
    else:
      logger.success("New dataset is better")
      return dataset2
  elif criteria.lower() == "mean":
    best_on_left: List[float] = []
    best_on_right: List[float] = []
    logger.info("Comparing datasets using mean criteria")
    for row in tqdm(range(len(dataset1)), desc="Iterating over rows"):
      best_texts_left: List[int] = dataset1[row]['best_texts']
      best_texts_right: List[int] = dataset2[row]['best_texts']
      if (-1 not in best_texts_left) and (-1 not in best_texts_right):
        logits_left: List[float] = [
            dataset1[row][f'best_logit_{id}'] for id in best_texts_left
        ]
        logits_right: List[float] = [
            dataset2[row][f'best_logit_{id}'] for id in best_texts_right
        ]
      else:
        logits_left = []
        logits_right = []
      mean_left = (sum(logits_left) /
                   len(logits_left)) if logits_left != [] else 0
      mean_right = (sum(logits_right) /
                    len(logits_right)) if logits_right != [] else 0
      best_on_left.append(mean_left)
      best_on_right.append(mean_right)
    wandb.log({"mean_logits": np.mean(best_on_right)})
    logger.success(f"Mean of previous: {np.mean(best_on_left)}")
    logger.success(f"Mean of actual: {np.mean(best_on_right)}")
    if np.median(best_on_left) > np.median(best_on_right):
      logger.success("Previous dataset is better")
      return dataset1
    else:
      logger.success("New dataset is better")
      return dataset2
  elif criteria.lower() == "meteor":
    logger.info("Comparing datasets using meteor criteria")
    if meteor_left != 0 and meteor_left > meteor_right:
      logger.success("Previous dataset is better")
      return dataset1
    else:
      logger.success("New dataset is better")
  else:
    raise ValueError("criteria must be 'sum','mean' or 'meteor'")


args = argument_parser()

logger.info("Loading metrics")
meteor = evaluate.load('meteor')
logger.success("Metrics loaded")

logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
num_new_tokens = tokenizer.add_special_tokens(
    {'additional_special_tokens': [args.separator]})
logger.success("Tokenizer loaded")

logger.info("Loading dataset")
original_dataset: Dataset = load_dataset()
logger.info("Adding useful columns to dataset")
original_dataset = original_dataset.map(
    batched_preprocess_function,
    batched=True,
    num_proc=10,
    load_from_cache_file=False,
    desc="Adding useful columns to dataset",
)
logger.success("Columns added")

if 'input_ids' in original_dataset.column_names:
  original_dataset.set_format("torch")
  logger.success("Dataset tokenized")
  best_tokenized_dataset = original_dataset
else:
  logger.info("Tokenizing dataset")
  tokenized_dataset: Dataset = original_dataset.map(
      batched_encode_function,
      batched=True,
      batch_size=args.batch_size,
      num_proc=4,
      fn_kwargs={'tokenizer': tokenizer},
      load_from_cache_file=False,
      desc="Tokenizing dataset",
  )
  tokenized_dataset.set_format("torch")
  logger.success("Dataset tokenized")
  best_tokenized_dataset = tokenized_dataset

if args.debug:
  logger.info("Debug mode")
  args.iterations = 3
  args.combinations = 2

tags = [
    f"{args.iterations}_iterations", f"{args.num_texts}_out_texts",
    f"{args.num_combinations}_combinations",
    f"{args.generated_texts}_generated_texts"
]
if args.lokr:
  tags.append("lokr")

run = wandb.init(project=f"adversarial_filtering", tags=tags)

for idx_iter in range(args.iterations):
  logger.info("Loading base model")
  base_model = AutoModelForMultipleChoice.from_pretrained(
      "severinsimmler/xlm-roberta-longformer-base-16384",
      torch_dtype=torch.bfloat16).to("cuda")
  logger.success(f"Base model loaded")

  logger.info("Resizing token embeddings of model")
  base_model.resize_token_embeddings(len(tokenizer))
  logger.info("Overwriting the embeddings to have better results")
  input_embeddings = base_model.get_input_embeddings().weight.data
  input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0,
                                                                 keepdim=True)
  input_embeddings[-num_new_tokens:] = input_embeddings_avg
  logger.success("Embeddings overwritten")

  target_modules: list[str] = [
      "query", "key", "value", "query_global", "key_global", "value_global",
      "classifier", "pooler"
  ]
  if args.lokr:
    logger.info("Creating Lokr model")
    config = LoKrConfig(
        r=8,
        alpha=8,
        target_modules=target_modules,
        inference_mode=False,
        use_effective_conv2d=True,
        modules_to_save=["classifier", "pooler"],
    )
    model = get_peft_model(base_model, config)
    logger.success(f"Lokr model created")
  else:
    model = base_model

  if args.debug:
    best_tokenized_dataset = best_tokenized_dataset.select(range(20))

  class_weight = torch.Tensor([1.0 / args.num_texts, 1.0] /
                              np.sum([1.0 / args.num_texts, 1.0]) * 2)

  logger.info("Splitting dataset")
  # 70% train, 30% test + validation
  train_testvalid: DatasetDict = best_tokenized_dataset.train_test_split(
      test_size=0.3, seed=np.random.randint(0, 1000))
  # Split the 30% test + valid in half test, half valid
  test_valid: DatasetDict = train_testvalid['test'].train_test_split(
      test_size=0.5, seed=np.random.randint(0, 1000))
  dataset: DatasetDict = DatasetDict({
      'train': train_testvalid['train'],
      'test': test_valid['test'],
      'val': test_valid['train']
  })
  logger.success("Dataset splitted")

  run = wandb.init(project=f"adversarial_filtering",
                   tags=tags,
                   name=f"af_{idx_iter}_of_{args.iterations}")

  # Define training arguments
  training_args = TrainingArguments(
      output_dir=
      f"af_outputs/{idx_iter}_of_{args.iterations}_with_{args.num_texts}_texts_and_{args.num_combinations}_comb",
      eval_strategy="epoch",
      save_strategy="epoch",
      overwrite_output_dir=True,
      learning_rate=5e-5,
      warmup_ratio=0.1,
      auto_find_batch_size=True,
      gradient_checkpointing=True,
      gradient_checkpointing_kwargs={"use_reentrant": False},
      bf16=True,
      group_by_length=True,
      bf16_full_eval=True,
      num_train_epochs=1,
      logging_steps=1,
      label_names=["labels"],
      remove_unused_columns=False,
      include_inputs_for_metrics=True,
      report_to="wandb",
  )

  trainer = WeightedTrainer(
      model=model,
      args=training_args,
      train_dataset=dataset["train"],
      eval_dataset=dataset["val"],
      tokenizer=tokenizer,
      data_collator=DataCollatorForMultipleChoice(
          tokenizer=tokenizer,
          num_choices=args.num_texts,
          idx=idx_iter,
      ),
      compute_metrics=compute_metrics,
      class_weights=class_weight,
  )

  # train model
  logger.info(f"Training model {idx_iter}")
  trainer.train()
  logger.info(f"Evaluating model {idx_iter}")
  metrics = trainer.evaluate(dataset["val"])
  wandb.log(metrics)
  logger.success(f"Model {idx_iter} evaluated")
  model_name = f"af_models/model_{idx_iter}_{args.iterations}_{args.num_texts}_{args.num_combinations}"
  if args.lokr:
    model.save_pretrained(model_name, save_embedding_layers=True)
  else:
    trainer.save_model(model_name)
  logger.success(f"Model {idx_iter} trained and saved")
  if args.lokr:
    logger.info("Loading base model and merging with PEFT model")
    model = trainer.model.base_model
    model = PeftModel.from_pretrained(
        model,
        model_name,
        torch_dtype=torch.float16,
        is_trainable=False,
    ).to('cuda')
    model.merge_and_unload()
    model.save_pretrained(model_name, save_embedding_layers=True)
    model.eval()
    logger.success(f"Model {idx_iter} merged and saved")
  else:
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
  new_tokenized_dataset = update_dataset(dataset)
  best_tokenized_dataset = best_dataset(best_tokenized_dataset,
                                        new_tokenized_dataset)
  best_tokenized_dataset.save_to_disk(
      f"data/af_out_{args.iterations}_{args.num_texts}_{args.num_combinations}"
  )
wandb.finish()
best_tokenized_dataset.remove_columns(["input_ids", "attention_mask"])
best_tokenized_dataset.save_to_disk(
    f"data/af_out_{args.iterations}_{args.num_texts}_{args.num_combinations}"
)
