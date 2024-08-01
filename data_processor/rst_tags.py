from collections import ChainMap
from typing import List, Set

RST_RELATIONS: Set = {
    'Topic-Change', 'Background', 'Contrast', 'Explanation', 'Comparison',
    'Temporal', 'Same-Unit', 'Enablement', 'Cause', 'Joint', 'Topic-Comment',
    'Manner-Means', 'Attribution', 'TextualOrganization', 'Evaluation',
    'Condition', 'Summary', 'Elaboration', 'span'
}


def generate_rst_tags(relation: str) -> dict:
  tags = {}
  for nc in ["N", "S"]:
    tags[f"{nc}_{relation}_initial_token"] = f"<{nc}:{relation}>"
    tags[f"{nc}_{relation}_final_token"] = f"<{nc}:{relation}>"
  return tags


RST_TAGS = dict(ChainMap(*list(map(generate_rst_tags, RST_RELATIONS))))
RST_TAGS_LIST = list(RST_TAGS.values())

RST_TAGS_COMPILED: List[str] = [
    '<N:Attribution>', '<S:Attribution>', '<N:Temporal>', '<S:Temporal>',
    '<N:Comparison>', '<S:Comparison>', '<N:Explanation>', '<S:Explanation>',
    '<N:Topic-Change>', '<S:Topic-Change>', '<N:Contrast>', '<S:Contrast>',
    '<N:Summary>', '<S:Summary>', '<N:span>', '<S:span>', '<N:Condition>',
    '<S:Condition>', '<N:Manner-Means>', '<S:Manner-Means>', '<N:Background>',
    '<S:Background>', '<N:Topic-Comment>', '<S:Topic-Comment>',
    '<N:TextualOrganization>', '<S:TextualOrganization>', '<N:Enablement>',
    '<S:Enablement>', '<N:Joint>', '<S:Joint>', '<N:Elaboration>',
    '<S:Elaboration>', '<N:Cause>', '<S:Cause>', '<N:Same-Unit>',
    '<S:Same-unit>', '<N:Evaluation>', '<S:Evaluation>'
]
