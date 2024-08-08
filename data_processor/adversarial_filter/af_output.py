import argparse
import os
from typing import Any, Dict, List
from uuid import uuid4

import datasets
from loguru import logger


def argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset_path',
      type=str,
      required=True,
  )
  parser.add_argument('--output_path', type=str, default=None)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
  )
  parser.add_argument('--num_proc', type=int, default=4)
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()
  return args


def best_samples(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
  best_fake_texts_ids = examples['best_texts']
  true_texts_ids = examples["ids_true"]
  batch_out = {
      "text_id": [],
      "text": [],
      "plot": [],
      "manipulation": [],
      "source_name": [],
      "recreated_from": [],
      "is_artificial": [],
      "label": [],
  }
  batch_size = len(best_fake_texts_ids)
  for idx_b in range(batch_size):
    if -1 not in best_fake_texts_ids[idx_b].tolist():
      for idx_t in best_fake_texts_ids[idx_b].tolist():
        # each fake text receives a unique id
        batch_out["text_id"].append(str(uuid4()))
        batch_out["text"].append(examples[f"text_{idx_t}"][idx_b])
        batch_out["plot"].append(examples[f"plot_{idx_t}"][idx_b])
        batch_out["manipulation"].append(
            examples[f"manipulation_{idx_t}"][idx_b])
        batch_out["source_name"].append(examples[f"dataset"][idx_b])
        batch_out["recreated_from"].append(examples["cluster_id"][idx_b])
        batch_out["is_artificial"].append(True)
        batch_out["label"].append(False)
    for idx_t in true_texts_ids[idx_b]:
      if examples[f"is_artificial_{idx_t}"][idx_b]:
        batch_out["text_id"].append(str(uuid4()))
        batch_out["is_artificial"].append(True)
      else:
        batch_out["text_id"].append(examples["cluster_id"][idx_b])
        batch_out["is_artificial"].append(False)
      batch_out["text"].append(examples[f"text_{idx_t}"][idx_b])
      batch_out["plot"].append(examples[f"plot_{idx_t}"][idx_b])
      batch_out["manipulation"].append([])
      batch_out["source_name"].append(examples[f"dataset"][idx_b])
      batch_out["recreated_from"].append(examples["cluster_id"][idx_b])
      batch_out["label"].append(True)
  return batch_out


args = argument_parser()
logger.info("Loading dataset from disk")
ds = datasets.load_from_disk(args.dataset_path)
logger.info("Dataset loaded")

if args.debug:
  args.num_proc = 1
  args.batch_size = 1

logger.info("Filtering adversarial examples")
ds_classification = ds.map(
    best_samples,
    batched=True,
    batch_size=args.batch_size,
    num_proc=args.num_proc,
    remove_columns=ds.column_names,
    desc="Filtering adversarial examples",
)
logger.success(f"Dataset filtered. New dataset size: {len(ds_classification)}")

if args.output_path is not None:
  ds_path_out = os.path.normpath(args.output_path) + "_af_output"
  if args.debug:
    ds_path_out = ds_path_out + "_debug"
else:
  ds_path_out = os.path.normpath(args.dataset_path)
  logger.info(f"Saving dataset to {ds_path_out}")
  if "tokenized_" in ds_path_out:
    ds_path_out = ds_path_out.replace("tokenized_", "")
  if "_af_input" in ds_path_out:
    ds_path_out = ds_path_out.replace("_af_input", "_af_output")
  else:
    ds_path_out = ds_path_out + "_af_output"

  if args.debug:
    ds_path_out = ds_path_out + "_debug"

logger.info(f"Saving dataset to {ds_path_out}")
ds_classification.save_to_disk(ds_path_out)
logger.success(f"Dataset saved to {ds_path_out}")
