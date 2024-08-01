from dataclasses import dataclass
from itertools import combinations
from random import choice, shuffle
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import DataCollatorWithPadding
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PaddingStrategy,
                                                  PreTrainedTokenizerBase)


def pad_without_fast_tokenizer_warning(
    tokenizer,
    *pad_args,
    **pad_kwargs,
) -> BatchEncoding:
  """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
  """

  # To avoid errors when using Feature extractors
  if not hasattr(tokenizer, "deprecation_warnings"):
    return tokenizer.pad(*pad_args, **pad_kwargs)

  # Save the state of the warning, then disable it
  warning_state = tokenizer.deprecation_warnings.get(
      "Asking-to-pad-a-fast-tokenizer", False)
  tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

  try:
    padded = tokenizer.pad(*pad_args, **pad_kwargs)
  finally:
    # Restore the state of the warning.
    tokenizer.deprecation_warnings[
        "Asking-to-pad-a-fast-tokenizer"] = warning_state

  return padded


@dataclass
class DataCollatorForMultipleChoice(DataCollatorWithPadding):
  """
  Data collator that will dynamically pad the inputs for multiple choice
  received.

  Args:
    tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]): The tokenizer used for encoding the data.\n
    padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):\n
      Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
      among:
      - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
        sequence is provided).
      - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
        acceptable input length for the model if that argument is not provided.
      - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).\n
    max_length (`int`, *optional*): Maximum length of the returned list and optionally padding length (see above).\n
    pad_to_multiple_of (`int`, *optional*): If set will pad the sequence to a multiple of the provided value. This is
    especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
  """
  tokenizer: PreTrainedTokenizerBase
  padding: Union[bool, str, PaddingStrategy] = True
  max_length: Optional[int] = None
  pad_to_multiple_of: Optional[int] = None
  num_choices: int = 3
  idx: int = 0

  def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size: int = len(features)
    texts_to_choose: List[int] = range(len(features[0]["input_ids"]))
    dropped_features: List[Dict[str, Any]] = []
    torch_columns = ['input_ids', 'attention_mask', 'labels']
    for idx_b in range(batch_size):
      labels_int: List[int] = features[idx_b]['labels'].to(torch.int).tolist()
      true_texts: List[int] = np.flatnonzero(labels_int).tolist()
      if "best_texts" in features[idx_b]:
        if -1 not in features[idx_b]["best_texts"].tolist():
          chosed_true = choice(true_texts)
          choosed_comb: List[int] = features[idx_b]["best_texts"].tolist()
          choosed_comb.append(chosed_true)
          shuffle(choosed_comb)
          choosed_comb = torch.tensor(choosed_comb, dtype=torch.int)
        else:
          false_texts: List[int] = list(set(texts_to_choose) - set(true_texts))
          all_combs: List[List[int]] = []
          for t_id in true_texts:
            combs: List[List[int]] = combinations(false_texts, self.num_choices)
            combs = [list(comb) + [t_id] for comb in combs]
            all_combs.extend(combs)
          # choose combination by idx
          shuffle(all_combs[self.idx])
          choosed_comb = torch.tensor(all_combs[self.idx], dtype=torch.int)
      else:
        false_texts: List[int] = list(set(texts_to_choose) - set(true_texts))
        all_combs: List[List[int]] = []
        for t_id in true_texts:
          combs: List[List[int]] = combinations(false_texts, self.num_choices)
          combs = [list(comb) + [t_id] for comb in combs]
          all_combs.extend(combs)
        # choose combination by idx
        shuffle(all_combs[self.idx])
        choosed_comb = torch.tensor(all_combs[self.idx], dtype=torch.int)
      model_input = {
          'input_ids':
              torch.index_select(features[idx_b]['input_ids'], 0, choosed_comb),
          'attention_mask':
              torch.index_select(
                  features[idx_b]['attention_mask'],
                  0,
                  choosed_comb,
              ),
          'labels':
              torch.index_select(features[idx_b]['labels'], 0, choosed_comb)
      }
      features[idx_b].update(model_input)
      all_keys = list(features[idx_b].keys())
      dropped_features.append({})
      for cl in all_keys:
        if cl not in torch_columns:
          dropped_features[idx_b][cl] = features[idx_b].pop(cl)

    # flatten features
    flattened_features: List[List[Dict[str, Any]]] = [[{
        k: v[i] for k, v in row.items()
    } for i in range(self.num_choices + 1)] for row in features]
    flattened_features = sum(flattened_features, [])
    # Pad the inputs
    batch: BatchEncoding = pad_without_fast_tokenizer_warning(
        self.tokenizer,
        flattened_features,
        padding=self.padding,
        max_length=self.max_length,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )
    # Un-flatten
    batch = {
        k: v.view(batch_size, self.num_choices + 1, -1)
        for k, v in batch.items()
    }
    # for i in range(batch_size):
    #   for k, v in dropped_features[i].items():
    #     if k not in batch:
    #       batch[k] = []
    #       batch[k].append(v)
    #     elif isinstance(v, list):
    #       batch[k].extend(v)
    #     else:
    #       batch[k].append(v)
    batch['labels'] = batch['labels'].view(batch_size, self.num_choices + 1)
    return batch
