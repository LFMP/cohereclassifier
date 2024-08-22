import warnings
from typing import Any, Dict, List

from torch import Tensor
from transformers import Pipeline
from transformers.pipelines.text_classification import (ClassificationFunction,
                                                        sigmoid, softmax)


class CustomPipelineForClassification(Pipeline):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __call__(self, inputs, **kwargs):
    """
        Classify the text(s) given as inputs.

        Args:
            inputs (`str` or `List[str]` or `Dict[str]`, or `List[Dict[str]]`):
                One or several texts to classify. In order to use text pairs for your classification, you can send a
                dictionary containing `{"text", "text_pair"}` keys, or a list of those.
            top_k (`int`, *optional*, defaults to `None`):
                How many results to return.
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                The function to apply to the model outputs in order to retrieve the scores. Accepts four different
                values:

                If this argument is not specified, then it will apply the following functions according to the number
                of labels:

                - If the model has a single label, will apply the sigmoid function on the output.
                - If the model has several labels, will apply the softmax function on the output.

                Possible values are:

                - `"sigmoid"`: Applies the sigmoid function on the output.
                - `"softmax"`: Applies the softmax function on the output.
                - `"none"`: Does not apply any function on the output.

        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:

            - **label** (`str`) -- The label predicted.
            - **score** (`float`) -- The corresponding probability.

            If `top_k` is used, one such dictionary is returned per label.
    """
    inputs = (inputs,)
    result = super().__call__(*inputs, **kwargs)
    _legacy = "top_k" not in kwargs
    if isinstance(inputs[0], str) and _legacy:
      # This pipeline is odd, and return a list when single item is run
      return [result]
    else:
      return result

  def _sanitize_parameters(self,
                           return_all_scores=None,
                           function_to_apply=None,
                           top_k="",
                           **tokenizer_kwargs):
    # Using "" as default argument because we're going to use `top_k=None` in user code to declare
    # "No top_k"
    preprocess_params = tokenizer_kwargs

    postprocess_params = {}
    if hasattr(self.model.config,
               "return_all_scores") and return_all_scores is None:
      return_all_scores = self.model.config.return_all_scores

    if isinstance(top_k, int) or top_k is None:
      postprocess_params["top_k"] = top_k
      postprocess_params["_legacy"] = False
    elif return_all_scores is not None:
      warnings.warn(
          "`return_all_scores` is now deprecated,  if want a similar functionality use `top_k=None` instead of"
          " `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`.",
          UserWarning,
      )
      if return_all_scores:
        postprocess_params["top_k"] = None
      else:
        postprocess_params["top_k"] = 1

    if isinstance(function_to_apply, str):
      function_to_apply = ClassificationFunction[function_to_apply.upper()]

    if function_to_apply is not None:
      postprocess_params["function_to_apply"] = function_to_apply
    return preprocess_params, {}, postprocess_params

  def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, Tensor]:
    if isinstance(inputs, dict):
      tokenized = self.tokenizer(**inputs,
                                 padding=True,
                                 truncation=True,
                                 return_tensors="pt",
                                 return_length=True,
                                 **tokenizer_kwargs)
    elif isinstance(inputs, list) and len(inputs) == 1 and isinstance(
        inputs[0], list) and len(inputs[0]) == 2:
      # It used to be valid to use a list of list of list for text pairs, keeping this path for BC
      tokenized = self.tokenizer(text=inputs[0][0],
                                 text_pair=inputs[0][1],
                                 padding=True,
                                 truncation=True,
                                 return_tensors="pt",
                                 return_length=True,
                                 **tokenizer_kwargs)
    elif isinstance(inputs, list):
      # This is likely an invalid usage of the pipeline attempting to pass text pairs.
      raise ValueError(
          "The pipeline received invalid inputs, if you are trying to send text pairs, you can try to send a"
          ' dictionary `{"text": "My text", "text_pair": "My pair"}` in order to send a text pair.'
      )
    else:
      tokenized = self.tokenizer(inputs,
                                 padding=True,
                                 truncation=True,
                                 return_tensors="pt",
                                 return_length=True,
                                 **tokenizer_kwargs)
    self.length = tokenized["length"]
    return tokenized

  def _forward(self, model_inputs):
    # `XXXForSequenceClassification` models should not use `use_cache=True` even
    # if it's supported
    model_inputs.pop("length", None)
    return self.model(**model_inputs)

  def postprocess(self,
                  model_outputs,
                  function_to_apply=None,
                  top_k=None,
                  _legacy=True) -> Dict[str, Any] | List[dict[str, Any]]:
    # `_legacy` is used to determine if we're running the naked pipeline and in backward
    # compatibility mode, or if running the pipeline with `pipeline(..., top_k=1)` we're running
    # the more natural result containing the list.
    # Default value before `set_parameters`
    if function_to_apply is None:
      if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
        function_to_apply = ClassificationFunction.SIGMOID
      elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
        function_to_apply = ClassificationFunction.SOFTMAX
      elif hasattr(self.model.config,
                   "function_to_apply") and function_to_apply is None:
        function_to_apply = self.model.config.function_to_apply
      else:
        function_to_apply = ClassificationFunction.NONE

    outputs = model_outputs["logits"][0]

    if self.framework == "pt":
      # To enable using fp16 and bf16
      outputs = outputs.float().numpy()
    else:
      outputs = outputs.numpy()

    if function_to_apply == ClassificationFunction.SIGMOID:
      scores = sigmoid(outputs)
    elif function_to_apply == ClassificationFunction.SOFTMAX:
      scores = softmax(outputs)
    elif function_to_apply == ClassificationFunction.NONE:
      scores = outputs
    else:
      raise ValueError(
          f"Unrecognized `function_to_apply` argument: {function_to_apply}")

    if top_k == 1 and _legacy:
      return {
          "label": self.model.config.id2label[scores.argmax().item()],
          "score": scores.max().item(),
          "length": self.length
      }

    dict_scores = [{
        "label": self.model.config.id2label[i],
        "score": score.item(),
        "length": self.length
    } for i, score in enumerate(scores)]
    # if not _legacy:
    #   dict_scores.sort(key=lambda x: x["score"], reverse=True)
    #   if top_k is not None:
    #     dict_scores = dict_scores[:top_k]
    return dict_scores
