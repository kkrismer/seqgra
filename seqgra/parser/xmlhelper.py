"""
MIT - CSAIL - Gifford Lab - seqgra

Implementation of static XML methods for parsers

@author: Konstantin Krismer
"""
from typing import Optional

class XMLHelper():
    @staticmethod
    def read_text_node(parent_node, node_name) -> Optional[str]:
        node = parent_node.getElementsByTagName(node_name)
        if len(node) == 0:
            return None
        elif node[0].firstChild is None:
            return ""
        else:
            return node[0].firstChild.nodeValue

    @staticmethod
    def read_immediate_text_node(node) -> str:
        if node.firstChild is None:
            return ""
        else:
            return node.firstChild.nodeValue

    @staticmethod
    def read_int_node(parent_node, node_name) -> int:
        node_value: Optional[str] = XMLHelper.read_text_node(parent_node, node_name)
        if node_value is None:
            return None
        else:
            return int(node_value)

    @staticmethod
    def read_float_node(parent_node, node_name) -> float:
        node_value: Optional[str] = XMLHelper.read_text_node(parent_node, node_name)
        if node_value is None:
            return None
        else:
            return float(node_value)
