from itertools import combinations
from random import shuffle
from typing import Any, Dict, List

import datasets
import numpy as np
import torch
from peft import LoKrConfig, TaskType, get_peft_model
from transformers import AutoModelForMultipleChoice, AutoTokenizer


def batched_process(examples: Dict[str, Any], num_texts: int = 3) -> None:
  batch_size: int = len(examples[list(examples.keys())[0]])
  texts_columns: List[str] = [clm for clm in examples.keys() if 'text_' in clm]
  texts_to_choose: List[int] = range(len(texts_columns))
  for idx_e in range(batch_size):
    true_texts = examples['ids_trues'][idx_e]
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
      model_input = {
          'input_ids':
              torch.index_select(examples['input_ids'][idx_e], 0, comb),
          'attention_mask':
              torch.index_select(examples['attention_mask'][idx_e], 0, comb),
          'labels':
              torch.index_select(examples['labels'][idx_e], 0, comb)
      }
      try:
        _ = model(**{
            k: v.unsqueeze(0).to("cuda") for k, v in model_input.items()
        })
      except:
        global count
        count += 1
        ids_to_remove.append(examples['cluster_id'][idx_e])
        if count % 100 == 0:
          print(f"Count: {count}")
          ds_removed = dataset.filter(
              lambda x: x['cluster_id'] in ids_to_remove,
              num_proc=4,
          )
          ds_removed.save_to_disk(
              "/home/luiz.pereira/cohereclassifier/data/tokenized_common_stories_manipulated_bad_samples"
          )
        break


separator = '||'

dataset = datasets.load_from_disk(
    "/home/luiz.pereira/cohereclassifier/data/tokenized_common_stories_manipulated_10_percent_af_input"
)

# include lenght based in input_ids
dataset = dataset.map(
    lambda x: {
        **x, "length": x["input_ids"].shape[1]
    },
    num_proc=10,
    desc="Adding length",
)
# order dataset by lenght
dataset = dataset.sort("length", reverse=True)

tokenizer = AutoTokenizer.from_pretrained(
    "severinsimmler/xlm-roberta-longformer-base-16384")
num_new_tokens = tokenizer.add_special_tokens(
    {'additional_special_tokens': [separator]})

model = AutoModelForMultipleChoice.from_pretrained(
    "severinsimmler/xlm-roberta-longformer-base-16384",
    # torch_dtype=torch.bfloat16
)

model.resize_token_embeddings(len(tokenizer))
input_embeddings = model.get_input_embeddings().weight.data
input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0,
                                                               keepdim=True)
input_embeddings[-num_new_tokens:] = input_embeddings_avg

model.to("cuda")

count = 0
ids_to_remove = []
dataset.map(batched_process, batched=True, batch_size=1)

ds_removed = dataset.filter(
    lambda x: x['cluster_id'] in ids_to_remove,
    num_proc=4,
)
ds_removed.save_to_disk(
    "/home/luiz.pereira/cohereclassifier/data/tokenized_common_stories_manipulated_bad_samples"
)

dataset = dataset.filter(
    lambda x: x['cluster_id'] not in ids_to_remove,
    num_proc=4,
)
print(f"Removed {len(ids_to_remove)} examples")

dataset.save_to_disk(
    "/home/luiz.pereira/cohereclassifier/data/tokenized_common_stories_manipulated_10_percent_af_input_filtered"
)
