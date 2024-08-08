from typing import Any, Dict, List

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict
from torch import Tensor

from .dmrst_parser.parser import DMRSTParser
from .pos_mix import POSMix
from .rst_mix import RSTMix


class CommonStories:

  def __init__(self, dataset: Dataset, batch_size: int = 400) -> None:
    self.dataset: Dataset = dataset
    self.batch_size: int = batch_size
    self.parser = DMRSTParser(
        model_path=
        "data_processor/dmrst_parser/checkpoint/multi_all_checkpoint.torchsave")
    self.rst_mixer = RSTMix()
    self.pos_mixer = POSMix()

  def parse_rst(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    infer_data = self.parser.inference(examples, field="text")
    removed = []
    for idx, data in enumerate(infer_data["text_edus"]):
      if len(data) <= 1:
        removed.append(idx)
    # remove from all keys
    for key in infer_data:
      infer_data[key] = [
          data for idx, data in enumerate(infer_data[key]) if idx not in removed
      ]
    return infer_data

  def rst_mix(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    new_texts = []
    texts = examples["text_rst"]
    edus = examples["text_edus"]
    for idx in range(len(texts)):
      new_texts.append(self.rst_mixer.process(texts[idx], edus[idx]))
    examples["text_rst_mixed"] = new_texts
    return examples

  def pos_mix(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    texts = examples["text"]
    examples['text_pos_mixed'] = self.pos_mixer.process(texts)
    return examples

  def process_dataset(self) -> DatasetDict:
    # get the ids from "recreated_from"
    unique_ids = list(set(self.dataset['recreated_from']))
    np.random.shuffle(unique_ids)
    # get the first 70% of the ids
    train_ids = unique_ids[:int(len(unique_ids) * 0.7)]
    # get the last 30% of the ids for validation and test
    test_valid_ids = unique_ids[int(len(unique_ids) * 0.7):]
    # split the 30% in half for test and validation
    test_ids = test_valid_ids[:int(len(test_valid_ids) * 0.5)]
    valid_ids = test_valid_ids[int(len(test_valid_ids) * 0.5):]

    ds_train = self.dataset.filter(
        lambda example: example['recreated_from'] in train_ids,
        num_proc=10,
        desc="Getting train data",
    )
    ds_val: Dataset = self.dataset.filter(
        lambda example: example['recreated_from'] in valid_ids,
        num_proc=10,
        desc="Getting validation data",
    )
    ds_test = self.dataset.filter(
        lambda example: example['recreated_from'] in test_ids,
        num_proc=10,
        desc="Getting test data",
    )
    dataset: DatasetDict = DatasetDict({
        'train': ds_train,
        'validation': ds_val,
        'test': ds_test,
    })

    for d in dataset:
      # cast the column with 'label' to ClassLabel with class_encode_column
      dataset[d] = dataset[d].cast_column("label", ClassLabel(names=["0", "1"]))
      dataset[d] = dataset[d].cast_column("is_artificial",
                                          ClassLabel(names=["0", "1"]))

    dataset = dataset.map(
        self.parse_rst,
        batched=True,
        batch_size=self.batch_size,
        desc="Parsing RST",
    )
    dataset = dataset.map(
        self.rst_mix,
        batched=True,
        batch_size=self.batch_size,
        desc="Mixing RST",
    )
    dataset = dataset.map(
        self.pos_mix,
        batched=True,
        batch_size=self.batch_size,
        desc="Mixing POS",
    )
    return dataset
