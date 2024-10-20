import argparse

import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--df", type=str, help="Path to the dataframe")
args = parser.parse_args()

df = pd.read_csv(args.df)
texts = df["text"].tolist()
translated = []
for text in tqdm(texts):
  try:
    translated.append(GoogleTranslator(source='en', target='pt').translate(text))
  except Exception as e:
    print(e)
    translated.append("ERROR")

df["text"] = translated
df = df[df["text"] != "ERROR"]
path = args.df.split(".")[0]
df.to_csv(f"{path}_translated.csv", index=False)
