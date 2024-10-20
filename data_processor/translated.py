import os
import uuid
from typing import Any, Dict, List

import datasets
import pandas as pd
from datasets import Dataset, DatasetDict

from .dmrst_parser.parser import DMRSTParser
from .pos_mix import POSMix
from .rst_mix import RSTMix


class Translated:

  def __init__(self, dataset: DatasetDict, batch_size: int = 400) -> None:
    self.ds = dataset
    self.batch_size = batch_size
    self.parser = DMRSTParser(
        model_path=
        "data_processor/dmrst_parser/checkpoint/multi_all_checkpoint.torchsave")
    self.rst_mixer = RSTMix()
    self.pos_mixer = POSMix(model="pt_core_news_lg")

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

  def process(self) -> DatasetDict:

    for split in ["train", "validation", "test"]:
      # cast the column 'label' to ClassLabel with class_encode_column
      self.ds[split] = self.ds[split].class_encode_column("label")

    dataset = self.ds.map(
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
    self.ds = dataset
    return dataset
