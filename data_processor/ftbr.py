import os
import uuid

import datasets
import pandas as pd
from datasets import Dataset, DatasetDict

from .dmrst_parser.parser import DMRSTParser
from .pos_mix import POSMix
from .rst_mix import RSTMix


class FakeTrueBr:

  def __init__(self, dataset_path: str, batch_size: int = 400) -> None:
    self.dataset_path = dataset_path
    self.batch_size = batch_size
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
    dfs = []
    for file in os.listdir(self.dataset_path):
      if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(self.dataset_path, file))
        new_data = []
        for _, row in df.iterrows():
          text_fake_id = str(uuid.uuid4())
          text_true_id = str(uuid.uuid4())
          new_data.append({
              "text_id": text_fake_id,
              "text": row["fake"],
              "link": row["link_f"],
              "label": 0,
              "related_to": text_true_id,
          })
          new_data.append({
              "text_id": text_fake_id,
              "text": row["true"],
              "link": row["link_t"],
              "label": 1,
              "related_to": text_fake_id,
          })
        new_df = pd.DataFrame(
            new_data,
            columns=["text_id", "text", "link", "label", "related_to"],
        )
        dfs.append(new_df)
    if len(dfs) == 1:
      ds = Dataset.from_pandas(dfs[0])
    else:
      ds = Dataset.from_pandas(pd.concat(dfs))

    # cast the column 'label' to ClassLabel with class_encode_column
    dataset = ds.class_encode_column("label")

    dataset = dataset.map(
        self.parse_rst,
        batched=True,
        batch_size=self.batch_size,
        desc="Parsing RST",
    )
    # get the rows without RST (text_rst equal to None)
    failed_texts = dataset.filter(
        lambda x: len(x["text_edus_breaks"]) < 3,
        desc="Filtering failed texts",
    )
    # get all compromised ids and relateds
    failed_ids = failed_texts["text_id"] + failed_texts["related_to"]
    # remove compromised from dataset
    dataset = dataset.filter(
        lambda x: x["text_id"] not in failed_ids,
        desc="Removing texts with failed RST parsing",
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
    # split the dataset into train and validation
    # 80% train, 10% test + validation
    train_testvalid = dataset.train_test_split(test_size=0.3,
                                               stratify_by_column="label")
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(
        test_size=0.5, stratify_by_column="label")
    # gather everyone if you want to have a single DatasetDict
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']
    })
    dataset = DatasetDict(dataset)
    return dataset
