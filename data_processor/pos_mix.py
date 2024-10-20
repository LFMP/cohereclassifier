import warnings

warnings.filterwarnings("ignore")
from dataclasses import dataclass

import spacy


@dataclass
class POSMix():
  """Class to mix POS info with text. Use `process(texts)` method to mix POS info with text
  """

  def __init__(self, model="en_core_web_trf") -> None:
    """Class to mix POS tags with text
    """
    spacy.require_gpu()
    spacy.prefer_gpu()
    self.nlp = spacy.load(model, disable=['ner', 'lemmatizer'])

  def process(self, texts: list[str]) -> list[str]:
    """Recive a list of texts and return a list with the text and the custom tags
    Args:
      examples (list[str]): list of texts to be processed

    Returns:
      list[str]: texts with the custom tags
    """
    new_texts = []
    for doc in self.nlp.pipe(texts):
      combined_tokes = []
      for token in doc:
        if token.pos_ != "SPACE" and token.pos_ != "PUNCT":
          combined_tokes.append(f"{token.text}_{token.tag_}")
      combined_text = " ".join(combined_tokes)
      new_texts.append(combined_text)
    return new_texts
