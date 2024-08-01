import torch
from transformers import AutoModel, AutoTokenizer

from .model_depth import ParsingNet


class DMRSTParser:

  def __init__(self,
               model_path: str = 'checkpoint/multi_all_checkpoint.torchsave'):

    bert_model = AutoModel.from_pretrained("xlm-roberta-base")

    bert_model = bert_model.cuda()

    for _, param in bert_model.named_parameters():
      param.requires_grad = False

    self.model_path = model_path

    self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base",
                                                   use_fast=True)

    self.model = ParsingNet(bert_model, bert_tokenizer=self.tokenizer)

    self.model = self.model.cuda()
    self.state_dict = torch.load(self.model_path)
    self.position_ids = self.state_dict.pop(
        'encoder.language_model.embeddings.position_ids')
    self.model.load_state_dict(self.state_dict)
    self.model = self.model.eval()

  def inference(self, examples: dict, field: str = "story") -> None:
    texts = examples[field]

    input_sentences = [
        self.tokenizer.tokenize(i, add_special_tokens=False) for i in texts
    ]
    with torch.no_grad():
      _, _, all_tree_parsing_pred, _, predict_EDU_breaks = self.model.TestingLoss(
          input_sentences,
          input_EDU_breaks=None,
          LabelIndex=None,
          ParsingIndex=None,
          GenerateTree=True,
          use_pred_segmentation=True)

    all_edus = []
    for idx, edu_x in enumerate(predict_EDU_breaks):
      current_edu = 0
      edus = ["" for _ in range(len(edu_x))]
      for i, element in enumerate(input_sentences[idx]):
        if i <= predict_EDU_breaks[idx][current_edu]:
          if i == 0:
            edus[current_edu] += element.replace("▁", "")
          elif element.find("▁") == -1:
            edus[current_edu] += element
          else:
            edus[current_edu] += " " + element.replace("▁", "")
        else:
          current_edu += 1
          edus[current_edu] += element.replace("▁", "")
      all_edus.append(edus)
    rst = sum(all_tree_parsing_pred, [])
    examples[f"{field}_edus"] = all_edus
    examples[f"{field}_edus_breaks"] = predict_EDU_breaks
    examples[f"{field}_rst"] = [r.strip().split() for r in rst]
    return examples
