import os

import datasets
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Value

from .dmrst_parser.parser import DMRSTParser
from .pos_mix import POSMix
from .rst_mix import RSTMix


class GCDC:

  def __init__(self,
               dataset_path: str,
               batch_size: int = 400,
               binary: bool = False) -> None:
    self.dataset_path = dataset_path
    self.batch_size = batch_size
    self.binary = binary
    self.parser = DMRSTParser(
        model_path=
        "data_processor/dmrst_parser/checkpoint/multi_all_checkpoint.torchsave")
    self.rst_mixer = RSTMix()
    self.pos_mixer = POSMix()

  def parse_rst(self, examples: dict):
    return self.parser.inference(examples, field="text")

  def rst_mix(self, examples):
    new_texts = []
    texts = examples["text_rst"]
    edus = examples["text_edus"]
    for idx in range(len(texts)):
      new_texts.append(self.rst_mixer.process(texts[idx], edus[idx]))
    examples["text_rst_mixed"] = new_texts
    return examples

  def pos_mix(self, examples):
    texts = examples["text"]
    examples['text_pos_mixed'] = self.pos_mixer.process(texts)
    return examples

  def load_dataset(self) -> DatasetDict:
    dataset = {}
    for file in os.listdir(self.dataset_path):
      if file.endswith(".jsonl"):
        df = pd.read_json(os.path.join(self.dataset_path, file), lines=True)
        ds = Dataset.from_pandas(df)
        dataset[file.split(".")[0]] = ds

    # create a column 'source_name' to store the source name of the dataset
    for d in dataset:
      dataset[d] = dataset[d].map(
          lambda _: {"source_name": d.split("_")[0]},
          num_proc=4,
          desc="Adding source name",
      )

    # cast text_id to string for all datasets
    for d in dataset:
      dataset[d] = dataset[d].cast_column("text_id", Value(dtype="string"))

    for d in dataset:
      dataset[d] = dataset[d].map(
          lambda e: {
              "label": e["label"] - 1,
              "label1": e["label1"] - 1,
              "label2": e["label2"] - 1,
              "label3": e["label3"] - 1,
          },
          remove_columns=["label", "label1", "label2", "label3"],
          num_proc=4,
          desc="Transforming labels to [0, 1, 2]",
      )
      # remove rows with label 1
      if self.binary:
        # actual gcdc labels are {0,1,2}, transformed to [0, 2]
        dataset[d] = dataset[d].filter(lambda e: e["label"] != 1)
        # tranform labels to [0, 1]
        dataset[d] = dataset[d].map(
            lambda e: {"label": 0 if e["label"] == 0 else 1},
            num_proc=4,
            desc="Transforming labels to [0, 1]",
        )
      # cast the column with 'label' to ClassLabel with class_encode_column
      dataset[d] = dataset[d].cast_column("label", ClassLabel(names=["0", "1"]))
      dataset[d] = dataset[d].cast_column("label1",
                                          ClassLabel(names=["0", "1"]))
      dataset[d] = dataset[d].cast_column("label2",
                                          ClassLabel(names=["0", "1"]))
      dataset[d] = dataset[d].cast_column("label3",
                                          ClassLabel(names=["0", "1"]))

    # concatenate all datasets by split
    ds = {}
    for split in ["train", "dev", "test"]:
      if split == "dev":
        ds["validation"] = datasets.concatenate_datasets([
            dataset[f"Clinton_{split}"],
            dataset[f"Yelp_{split}"],
            dataset[f"Enron_{split}"],
            dataset[f"Yahoo_{split}"],
        ])
      else:
        ds[split] = datasets.concatenate_datasets([
            dataset[f"Clinton_{split}"],
            dataset[f"Yelp_{split}"],
            dataset[f"Enron_{split}"],
            dataset[f"Yahoo_{split}"],
        ])
    dataset = DatasetDict(ds)
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
