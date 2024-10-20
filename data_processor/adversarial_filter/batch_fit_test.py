import argparse
import os
from itertools import combinations
from random import shuffle
from typing import Any, Dict, List

import datasets
import numpy as np
import torch
from processor import batched_encode_function
from transformers import AutoModelForMultipleChoice, AutoTokenizer


def batched_process(examples: Dict[str, Any], num_texts: int = 3) -> None:
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
      combs: List[List[int]] = combinations(false_texts, num_texts)
      combs = [
          np.random.permutation(list(comb) + [idx_t.item()]) for comb in combs
      ]
      all_combs.extend(combs)
    shuffle(all_combs)
    all_combs = all_combs[:5]

    for comb in all_combs:
      # get tensors using comb as index
      comb = torch.tensor(comb, dtype=torch.int)
      labels = torch.tensor(
          [examples[f"label_{i}"][idx_e] for i in range(len(texts_columns))],
          dtype=torch.float)
      model_input = {
          'input_ids':
              torch.index_select(examples['input_ids'][idx_e], 0, comb),
          'attention_mask':
              torch.index_select(examples['attention_mask'][idx_e], 0, comb),
          'labels':
              torch.index_select(labels, 0, comb)
      }
      try:
        _ = model(**{
            k: v.unsqueeze(0).to("cuda") for k, v in model_input.items()
        })
      except Exception as e:
        global count
        count += 1
        ids_to_remove.append(examples['cluster_id'][idx_e])
        len_removed.extend(examples['length'])
        if count % 1000 == 0:
          print(f"Count: {count}")
          print(f"Median length of removed texts: {np.median(len_removed)}")
          ds_removed = dataset.filter(
              lambda x: x['cluster_id'] in ids_to_remove,
              num_proc=4,
          )
          ds_removed.save_to_disk(
              f"processed/tokenized_{dataset_name}_bad_samples")
        break


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
args = parser.parse_args()
separator = '||'

normalized_dataset_path = os.path.normpath(args.dataset_path)
dataset_name = os.path.basename(normalized_dataset_path)
dataset = datasets.load_from_disk(args.dataset_path)

tokenizer = AutoTokenizer.from_pretrained(
    "severinsimmler/xlm-roberta-longformer-base-16384")
num_new_tokens = tokenizer.add_special_tokens(
    {'additional_special_tokens': [separator]})

tokenized_dataset = dataset.map(
    batched_encode_function,
    batched=True,
    num_proc=4,
    fn_kwargs={'tokenizer': tokenizer},
    desc="Tokenizing dataset",
)
tokenized_dataset.set_format("torch")
# include lenght based in input_ids
tokenized_dataset = tokenized_dataset.map(
    lambda x: {
        **x, "length": x["input_ids"].shape[1]
    },
    num_proc=10,
    desc="Adding length",
)
# order tokenized_dataset by lenght
tokenized_dataset = tokenized_dataset.sort("length", reverse=True)
tokenized_dataset.save_to_disk(f"processed/tokenized_{dataset_name}")

model = AutoModelForMultipleChoice.from_pretrained(
    "severinsimmler/xlm-roberta-longformer-base-16384",
    torch_dtype=torch.bfloat16)

model.resize_token_embeddings(len(tokenizer))
input_embeddings = model.get_input_embeddings().weight.data
input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0,
                                                               keepdim=True)
input_embeddings[-num_new_tokens:] = input_embeddings_avg

model.to("cuda")

count = 0
ids_to_remove = []
len_removed = []
tokenized_dataset.map(batched_process, batched=True, batch_size=1)

ds_removed = tokenized_dataset.filter(
    lambda x: x['cluster_id'] in ids_to_remove,
    num_proc=4,
)
ds_removed.save_to_disk(f"processed/tokenized_{dataset_name}_bad_samples")

tokenized_dataset = tokenized_dataset.filter(
    lambda x: x['cluster_id'] not in ids_to_remove,
    num_proc=4,
)
print(f"Removed {len(ids_to_remove)} examples")

tokenized_dataset.save_to_disk(f"processed/tokenized_{dataset_name}_filtered")
