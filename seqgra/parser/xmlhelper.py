"""
MIT - CSAIL - Gifford Lab - seqgra

Implementation of static XML methods for parsers

@author: Konstantin Krismer
"""


class XMLHelper():
    @staticmethod
    def read_text_node(parent_node, node_name) -> str:
        node = parent_node.getElementsByTagName(node_name)
        if len(node) == 0:
            return ""
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
        node_value: str = XMLHelper.read_text_node(parent_node, node_name)
        return int(node_value)

    @staticmethod
    def read_float_node(parent_node, node_name) -> float:
        node_value: str = XMLHelper.read_text_node(parent_node, node_name)
        return float(node_value)
