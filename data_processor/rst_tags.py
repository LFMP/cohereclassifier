from collections import ChainMap
from typing import List, Set

RST_RELATIONS: Set = {
    'Topic-Change', 'Background', 'Contrast', 'Explanation', 'Comparison',
    'Temporal', 'Same-Unit', 'Enablement', 'Cause', 'Joint', 'Topic-Comment',
    'Manner-Means', 'Attribution', 'TextualOrganization', 'Evaluation',
    'Condition', 'Summary', 'Elaboration', 'span'
}

RelationTable = [
    'Attribution_SN', 'Enablement_NS', 'Cause_SN', 'Cause_NN', 'Temporal_SN',
    'Condition_NN', 'Cause_NS', 'Elaboration_NS', 'Background_NS',
    'Topic-Comment_SN', 'Elaboration_SN', 'Evaluation_SN', 'Explanation_NN',
    'TextualOrganization_NN', 'Background_SN', 'Contrast_NN', 'Evaluation_NS',
    'Topic-Comment_NN', 'Condition_NS', 'Comparison_NS', 'Explanation_SN',
    'Contrast_NS', 'Comparison_SN', 'Condition_SN', 'Summary_SN',
    'Explanation_NS', 'Enablement_SN', 'Temporal_NN', 'Temporal_NS',
    'Topic-Comment_NS', 'Manner-Means_NS', 'Same-Unit_NN', 'Summary_NS',
    'Contrast_SN', 'Attribution_NS', 'Manner-Means_SN', 'Joint_NN',
    'Comparison_NN', 'Evaluation_NN', 'Topic-Change_NN', 'Topic-Change_NS',
    'Summary_NN', 'span_NS'
]


def generate_rst_tags(relation: str) -> dict:
  tags = {}
  for rt in RelationTable:
    relation, nuclearity = rt.split('_')
    for nc in nuclearity:
      tags[f"{nc}_{relation}_token"] = f"<{nc}:{relation}>"
  return tags


RST_TAGS = dict(ChainMap(*list(map(generate_rst_tags, RST_RELATIONS))))
RST_TAGS_LIST = list(RST_TAGS.values())

RST_TAGS_COMPILED: List[str] = sorted([
    '<N:Attribution>', '<S:Attribution>', '<N:Temporal>', '<S:Temporal>',
    '<N:Comparison>', '<S:Comparison>', '<N:Explanation>', '<S:Explanation>',
    '<N:Topic-Change>', '<S:Topic-Change>', '<N:Contrast>', '<S:Contrast>',
    '<N:Summary>', '<S:Summary>', '<N:span>', '<S:span>', '<N:Condition>',
    '<S:Condition>', '<N:Manner-Means>', '<S:Manner-Means>', '<N:Background>',
    '<S:Background>', '<N:Topic-Comment>', '<S:Topic-Comment>',
    '<N:TextualOrganization>', '<N:Enablement>', '<S:Enablement>', '<N:Joint>',
    '<N:Elaboration>', '<S:Elaboration>', '<N:Cause>', '<S:Cause>',
    '<N:Same-Unit>', '<N:Evaluation>', '<S:Evaluation>'
])
