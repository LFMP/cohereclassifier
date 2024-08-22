import argparse
import os
from typing import Any, Dict

import datasets
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from custom_pipeline import CustomPipelineForClassification
from data_processor.pos_tags import POS_TAGS_COMPILED
from data_processor.rst_tags import RST_TAGS_COMPILED

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_path',
    type=str,
    required=True,
)
parser.add_argument('--tokenizer',
                    type=str,
                    default="severinsimmler/xlm-roberta-longformer-base-16384")
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--prefix', type=str, default='gcdc')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


def inference(examples,
              model_name: str,
              text_field: str = "text") -> Dict[str, Any]:
  results = classifier(inputs=examples[text_field])
  info = []
  for result in results:
    info.append([(result[0]['score'], result[1]['score']), result[0]['length']])
  examples[f"{model_name}"] = info
  return examples


ds = datasets.load_from_disk(args.dataset_path)

checkpoints = sorted(
    [f"checkpoints/{c}" for c in os.listdir("checkpoints") if args.prefix in c])

for idx, c in enumerate(checkpoints):
  # load model
  model_name = c.replace("checkpoints/", "")
  logger.info(f"Loading model {idx} of {len(checkpoints)} from {c}")
  model = AutoModelForSequenceClassification.from_pretrained(
      c, local_files_only=True).to("cuda")
  model.eval()
  logger.success(f"Model loaded from {c}")
  if "rst-True" in c:
    rst = True
    pos = False
  elif "pos-True" in c:
    rst = False
    pos = True
  else:
    rst = False
    pos = False

  if "text" in ds["train"].column_names:
    text_field = "text"
  elif "story" in ds["train"].column_names:
    text_field = "story"
  else:
    raise ValueError(
        "Dataset does not have a default column field ('text' or 'story')")

  logger.info("Loading tokenizer")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
  if rst:
    logger.info("Adding RST tags to tokenizer")
    num_new_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": RST_TAGS_COMPILED},
        replace_additional_special_tokens=False,
    )
  elif pos:
    logger.info("Adding POS tags to tokenizer")
    num_new_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": POS_TAGS_COMPILED},
        replace_additional_special_tokens=False,
    )

  logger.info("Tokenizer loaded")

  if rst:
    text_field = f"{text_field}_rst_mixed"
  elif pos:
    text_field = f"{text_field}_pos_mixed"
  else:
    text_field = text_field

  classifier = CustomPipelineForClassification(
      model=model,
      tokenizer=tokenizer,
      top_k=None,
      framework="pt",
      torch_dtype=torch.bfloat16,
      device=0,
  )
  ds["validation"] = ds["validation"].map(
      inference,
      batched=True,
      batch_size=args.batch_size,
      fn_kwargs={
          "text_field": text_field,
          "model_name": model_name
      },
      desc=f"Running inference",
  )

dataset_name = os.path.basename(os.path.normpath(args.dataset_path)).lower()
ds.save_to_disk(f"data/{args.prefix}_in_{dataset_name}_scores")
