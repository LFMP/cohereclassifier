from dataclasses import dataclass


@dataclass
class RSTMix():
  """Class to mix RST units with EDUs. Use `process(rst_str, edus)` method to mix RST units with EDUs
  """

  def __init__(self) -> None:
    """Class to mix RST units with EDUs
    """
    pass

  def block_process(self, block: str) -> list:
    """Recive a block of RST and return a list with the init, end, nuclearity and relation of the block

    Args:
      block (str): A block of RST

    Returns:
      list: A list with the init, end, nuclearity and relation of the block
    """
    new_block = block.replace("(", "")
    new_block = new_block.replace(")", "")
    fp, sp = new_block.split("=")
    span_init, span_nuclearity = fp.split(":")
    span_relation, span_end = sp.split(":")
    return [span_init, span_end, span_nuclearity, span_relation]

  def unit2dict(self,
                rst_units: list) -> list[dict[str, dict[str, (str | int)]]]:
    """Recive a list of RST units and return a list of dicts with the init, end, nuclearity and relation of the block

    Args:
      rst_units (list): list of RST units

    Returns:
      _type_: list of dicts with the init, end, nuclearity and relation of the block
    """
    rst_dict = []
    for unit in rst_units:
      first_block, second_block = unit.split(",")
      fb_init, fb_end, fb_nuclearity, fb_relation = self.block_process(
          first_block)
      sb_init, sb_end, sb_nuclearity, sb_relation = self.block_process(
          second_block)
      unit_dict = {
          "first_block": {
              "init": int(fb_init) - 1,
              "end": int(fb_end) - 1,
              "nuclearity": fb_nuclearity,
              "relation": fb_relation
          },
          "second_block": {
              "init": int(sb_init) - 1,
              "end": int(sb_end) - 1,
              "nuclearity": sb_nuclearity,
              "relation": sb_relation
          }
      }
      rst_dict.append(unit_dict)
    return rst_dict

  def dict2tag(self, relation_unit: dict) -> list:
    """Recive a dict with the init, end, nuclearity and relation of the block and return a list of custom html tags

    Args:
      relation_unit (dict): dict with the init, end, nuclearity and relation of the block

    Returns:
        list: list of custom html tags
    """
    fb_nuclearity = relation_unit["first_block"]["nuclearity"][0]
    sb_nuclearity = relation_unit["second_block"]["nuclearity"][0]
    fb_relation = relation_unit["first_block"]["relation"]
    sb_relation = relation_unit["second_block"]["relation"]

    fb_init_tag = f'<{fb_nuclearity}:{fb_relation}>'
    fb_end_tag = f'<{fb_nuclearity}:{fb_relation}>'

    sb_init_tag = f'<{sb_nuclearity}:{sb_relation}>'
    sb_end_tag = f'<{sb_nuclearity}:{sb_relation}>'

    return [(relation_unit["first_block"]["init"], fb_init_tag),
            (relation_unit["first_block"]["end"], fb_end_tag),
            (relation_unit["second_block"]["init"], sb_init_tag),
            (relation_unit["second_block"]["end"], sb_end_tag)]

  def mix_rst_with_text(self, rst_dicts: list, edus: list) -> str:
    """Recive a list of dicts with the init, end, nuclearity and relation of the block and return a string with the text and the custom html tags

    Args:
      rst_dicts (list): list of dicts with the init, end, nuclearity and relation of the block
      edus (list): list of EDUs

    Returns:
      str: string with the text and the custom tags
    """
    new_edus = edus
    new_order_dicts = rst_dicts
    for relation in new_order_dicts:
      data = self.dict2tag(relation)
      new_edus[data[0][0]] = f"{data[0][1]}{new_edus[data[0][0]]}"
      new_edus[data[2][0]] = f"{data[2][1]}{new_edus[data[2][0]]}"
      new_edus[data[1][0]] = f"{new_edus[data[1][0]]}{data[1][1]}"
      new_edus[data[3][0]] = f"{new_edus[data[3][0]]}{data[3][1]}"
    return ''.join(new_edus)

  def process(self, rst_units: list[str], edus: list) -> str:
    """Recive a list of RST units and a list of EDUs and return a string with the text and the custom html tags

    Args:
      rst_str (str): string of RST units
      edus (list): list of EDUs

    Returns:
      str: string with the text and the custom html tags
    """
    rst_dicts = self.unit2dict(rst_units=rst_units)
    return self.mix_rst_with_text(rst_dicts=rst_dicts, edus=edus)
