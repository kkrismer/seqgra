"""Gifford Lab - seq-grammar
Parser for XML configuration files (using Strategy design pattern)

@author: Konstantin Krismer
"""

from typing import List

from abc import ABC, abstractmethod
from parser import Parser
from model.background import Background
from model.datageneration import DataGeneration
from model.condition import Condition
from model.sequenceelement import SequenceElement

class XMLParser(Parser):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    def __init__(self, config: str) -> None:
        self._dom = xml.dom.minidom.parseString(config)
        self._general_element = self._dom.getElementsByTagName("general")[0]

    @staticmethod
    def __read_text_node(parent_node, node_name):
        node = parent_node.getElementsByTagName(node_name)
        if len(node) == 0:
            return ""
        else if node[0].firstChild is None:
            return ""
        else:
            return node[0].firstChild.nodeValue
    
    @staticmethod
    def __read_int_node(parent_node, node_name):
        node_value = XMLParser.__read_text_node(parent_node, node_name)
        return int(node_value)
    
    @staticmethod
    def __read_float_node(parent_node, node_name):
        node_value = XMLParser.__read_text_node(parent_node, node_name)
        return float(node_value)

    def get_id(self) -> str:
        return self._general_element.getAttribute("id")
    
    def get_label(self) -> str:
        return XMLParser.__read_text_node(self._general_element, "label")
    
    def get_description(self) -> str:
        return XMLParser.__read_text_node(self._general_element, "description")
    
    def get_sequence_space(self) -> str:
        return XMLParser.__read_text_node(self._general_element, "sequencespace")

    def get_background(self) -> Background:
        background_element = self._dom.getElementsByTagName("background")[0]
        min_length = XMLParser.__read_int_node(background_element, "minlength")
        max_length = XMLParser.__read_int_node(background_element, "maxlength")

        distribution_elements = background_element.getElementsByTagName("alphabetdistribution")]       
        distributions = [XMLParser.__parse_alphabet_distribution(distribution_element) for distribution_element in distribution_elements]
        
        self.background = Background(min_length, max_length, distributions)
    
    @staticmethod
    def __parse_alphabet_distribution(alphabet_distribution_element):
        if alphabet_distribution_element.hasAttribute("condition"):
            condition = alphabet_distribution_element.getAttribute("condition")
        else:
            condition = None

        letter_elements = alphabet_distribution_element.getElementsByTagName("letter")]
        letters = [XMLParser.__parse_letter(letter_element) for letter_element in letter_elements]
        
        return AlphabetDistribution(letters, condition)
    
    @staticmethod
    def __parse_letter(letter_element):
        return Letter(letter_element.firstChild.nodeValue,
                      letter_element.getAttribute("probability"))

    def get_data_generation(self) -> DataGeneration:
        data_generation_element = self._dom.getElementsByTagName("datageneration")[0]
    
    def get_conditions(self) -> List[Condition]:
        conditions_element = self._dom.getElementsByTagName("conditions")[0]
    
    def get_sequence_elements(self) -> List[SequenceElement]:
        sequence_elements_element = self._dom.getElementsByTagName("sequenceelements")[0]
