from typing import Any, Dict, List

import numpy as np
import torch


def batched_preprocess_function(examples: Dict[str, Any],
                                num_texts: int = 3) -> Dict[str, Any]:
  # gather the labels
  labels_columns: List[str] = [
      clm for clm in examples.keys() if 'label_' in clm
  ]
  if len(labels_columns) != 0:
    labels: List[List[float]] = [[
        float(examples[clm][idx]) for clm in labels_columns
    ] for idx in range(len(examples[labels_columns[0]]))]
  else:
    labels = examples['labels'].tolist()
  # get the original plots
  texts_columns: List[str] = [clm for clm in examples.keys() if 'text_' in clm]
  ids_true: List[List[int]] = [np.flatnonzero(lb).tolist() for lb in labels]
  if 'ids_trues' in examples:
    examples.pop('ids_trues')
  if 'best_logit_0' in examples:
    for idx in range(len(texts_columns)):
      examples.pop(f'best_logit_{idx}')
  examples['ids_true'] = ids_true
  examples['best_texts'] = torch.tensor([[-1] * num_texts] * len(ids_true),
                                        dtype=torch.int)
  examples['labels'] = labels
  for l in labels_columns:
    examples.pop(l)
  for idx_t in range(len(texts_columns)):
    examples[f'best_logit_{idx_t}'] = torch.tensor([0] * len(ids_true),
                                                   dtype=torch.float)
  return examples


def batched_encode_function(examples: Dict[str, Any],
                            tokenizer) -> Dict[str, Any]:
  # gather the texts
  texts_columns: List[str] = [clm for clm in examples.keys() if 'text_' in clm]
  label_columns: List[str] = [clm for clm in examples.keys() if 'label_' in clm]
  if len(label_columns) != 0:
    labels: List[List[float]] = [[
        float(examples[clm][idx]) for clm in label_columns
    ] for idx in range(len(examples[label_columns[0]]))]
  else:
    labels = examples['labels'].tolist()
  ids_true: List[List[int]] = [np.flatnonzero(lb).tolist() for lb in labels]
  examples['ids_true'] = ids_true
  true_texts: List[int] = [example[0] for example in examples['ids_true']]
  original_plots: List[str] = [
      examples[f'plot_{id}'][idx] for idx, id in enumerate(true_texts)
  ]

  # make a copy of plot for each text * (len(texts_columns))
  headers: List[List[str]] = [
      [plot] * len(texts_columns) for plot in original_plots
  ]
  texts: List[List[str]] = [[examples[clm][idx]
                             for clm in texts_columns]
                            for idx in range(len(true_texts))]

  tokenized_examples = []
  for idx in range(len(texts)):
    tokenized_example = tokenizer(
        headers[idx],
        texts[idx],
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_length=True,
    )
    tokenized_examples.append(tokenized_example)
  for example in tokenized_examples:
    for k, v in example.items():
      if k not in examples:
        examples[k] = []
      examples[k].append(v)

  return examples
