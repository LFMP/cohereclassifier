import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from model_depth import ParsingNet

argparser = argparse.ArgumentParser()
argparser.add_argument("--base_dir", type=str, default="data/Manplts")
argparser.add_argument("--output_dir", type=str, default="data/Manplts")
args = argparser.parse_args()

def inference(model, tokenizer, input_sentences, batch_size=1):
  LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

  input_sentences = [
      tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences
  ]
  all_segmentation_pred = []
  all_tree_parsing_pred = []
  with torch.no_grad():
    for loop in range(LoopNeeded):
      StartPosition = loop * batch_size
      EndPosition = (loop + 1) * batch_size
      if EndPosition > len(input_sentences):
        EndPosition = len(input_sentences)

      input_sen_batch = input_sentences[StartPosition:EndPosition]
      _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(
          input_sen_batch,
          input_EDU_breaks=None,
          LabelIndex=None,
          ParsingIndex=None,
          GenerateTree=True,
          use_pred_segmentation=True)
      all_segmentation_pred.extend(predict_EDU_breaks)
      all_tree_parsing_pred.extend(SPAN_batch)
  return input_sentences, all_segmentation_pred, all_tree_parsing_pred


def process_Manplts(base_dir, output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  df_list = []
  id = 0
  for f_name in tqdm(os.listdir(base_dir)):
    if f_name.endswith(".tsv"):
      df = pd.read_table(os.path.join(base_dir, f_name))
      df["story"] = df["story"].str.replace(r'<[^<>]*>', '', regex=True)
      ids = []
      for i in range(len(df)):
        if i % 4 == 0:
          id += 1
        ids.append(id)
      df['story_n'] = ids
      df['split'] = [f"{os.path.basename(base_dir).split('_')[-1].replace('tsv','')}" * len(df)]
      df_list.append(df)

  df = pd.concat(df_list, ignore_index=True)
  df = df.reset_index(drop=True)
  tail = os.path.split(base_dir)[1]
  fname = f'{tail}_processed.csv'
  df.to_csv(os.path.join(output_dir, fname), index=False)

def extract(df, output_dir):
  tail = os.path.split(base_dir)[1]
  fname = f'{tail}_processed.csv'
  model_path = 'checkpoint/multi_all_checkpoint.torchsave'
  batch_size = 1

  """ BERT tokenizer and model """
  bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base",
                                                use_fast=True)
  bert_model = AutoModel.from_pretrained("xlm-roberta-base")

  bert_model = bert_model.cuda()

  for _, param in bert_model.named_parameters():
    param.requires_grad = False

  model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)

  model = model.cuda()
  model.load_state_dict(torch.load(model_path))
  model = model.eval()

  json_list = []
  
  for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    current_edu = 0
    sentence = []
    sentence.append(row['story'])
    input_sentences, all_segmentation_pred, all_tree_parsing_pred = inference(model, bert_tokenizer, sentence, batch_size)

    edus = ["" for _ in range(len(all_segmentation_pred[0]))]

    for i, element in enumerate(input_sentences[0]):
      if i <= all_segmentation_pred[0][current_edu]:
        if i == 0:
          edus[current_edu] += element.replace("▁", "")
        elif element.find("▁") == -1:
          edus[current_edu] += element
        else:
          edus[current_edu] += " " + element.replace("▁", "")
      else:
        current_edu += 1
        edus[current_edu] += element.replace("▁", "")

    dataset_unit = {}
    dataset_unit["label"] = row['label']
    dataset_unit["id"] = row['story_n']
    dataset_unit["edus"] = edus
    dataset_unit["edus_breaks"] = all_segmentation_pred[0]
    dataset_unit["rst"] = all_tree_parsing_pred[0][0]
    dataset_unit["text"] = ' '.join(edus)
    json_list.append(dataset_unit)
    if idx % 100 == 0:
      df_out = pd.DataFrame(json_list)
      df_out.to_csv(os.path.join(output_dir, fname), index=False)
  
  df_out = pd.DataFrame(json_list)
  df_out.to_csv(os.path.join(output_dir, fname), index=False)
    
if __name__ == "__main__":
  base_dir = args.base_dir
  output_dir = args.output_dir
  print(f"Cleaning...")
  process_Manplts(base_dir, output_dir)
  df = pd.read_csv(os.path.join(output_dir, f"{os.path.split(base_dir)[1]}_processed.csv"))
  print("Extracting...")
  extract(df, output_dir)
