POS_TAGS_COMPILED: list[str] = sorted([
    '_SCONJ', '_NUM', '_X', '_NOUN', '_AUX', '_ADP', '_DET', '_VERB', '_SYM',
    '_PROPN', '_ADV', '_ADJ', '_PRON', '_CCONJ', '_PART', '_INTJ'
])


def get_pos_tags(model: str = "en_core_web_trf"):
  import spacy
  nlp = spacy.load(
      model, disable=['transformer', 'tagger', 'parser', 'lemmatizer', 'ner'])

  patterns = nlp.get_pipe('attribute_ruler').patterns
  pos_set = set()

  for pattern in patterns:
    attrs: dict = pattern['attrs']
    pos_tag = attrs.get('TAG', None)
    if pos_tag is not None:
      pos_set.add(pos_tag)

  pos_list = list(pos_set)
  POS_TAGS = sorted(
      [f"_{tag} " for tag in pos_list if tag != "SPACE" and tag != "PUNCT"],)
  return POS_TAGS
