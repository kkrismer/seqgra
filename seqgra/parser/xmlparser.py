"""Gifford Lab - seq-grammar
Parser for XML configuration files (using Strategy design pattern)

@author: Konstantin Krismer
"""

from xml.dom.minidom import Document, parseString

from typing import Any, List, Tuple
from abc import ABC, abstractmethod

from seqgra.parser.parser import Parser
from seqgra.model.background import Background
from seqgra.model.datageneration import DataGeneration
from seqgra.model.condition import Condition
from seqgra.model.sequenceelement import SequenceElement, KmerBasedSequenceElement, MatrixBasedSequenceElement
from seqgra.model.alphabetdistribution import AlphabetDistribution
from seqgra.model.rule import Rule
from seqgra.model.spacingconstraint import SpacingConstraint

class XMLParser(Parser):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    def __init__(self, config: str) -> None:
        self._dom: Document = parseString(config)
        self._general_element: Any = self._dom.getElementsByTagName("general")[0]

    @staticmethod
    def __read_text_node(parent_node, node_name) -> str:
        node: Any = parent_node.getElementsByTagName(node_name)
        if len(node) == 0:
            return ""
        elif node[0].firstChild is None:
            return ""
        else:
            return node[0].firstChild.nodeValue
    
    @staticmethod
    def __read_int_node(parent_node, node_name) -> int:
        node_value: str = XMLParser.__read_text_node(parent_node, node_name)
        return int(node_value)
    
    @staticmethod
    def __read_float_node(parent_node, node_name) -> float:
        node_value: str = XMLParser.__read_text_node(parent_node, node_name)
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
        background_element: Any = self._dom.getElementsByTagName("background")[0]
        min_length: int = XMLParser.__read_int_node(background_element, "minlength")
        max_length: int = XMLParser.__read_int_node(background_element, "maxlength")

        distribution_elements: Any = background_element.getElementsByTagName("alphabetdistribution")
        distributions: List[AlphabetDistribution] = [XMLParser.__parse_alphabet_distribution(distribution_element) for distribution_element in distribution_elements]
        
        self.background: Background = Background(min_length, max_length, distributions)
    
    @staticmethod
    def __parse_alphabet_distribution(alphabet_distribution_element) -> AlphabetDistribution:
        if alphabet_distribution_element.hasAttribute("condition"):
            condition: str = alphabet_distribution_element.getAttribute("condition")
        else:
            condition: str = None

        letter_elements: Any = alphabet_distribution_element.getElementsByTagName("letter")
        letters: List[Tuple[str, float]] = [XMLParser.__parse_letter(letter_element) for letter_element in letter_elements]
        
        return AlphabetDistribution(letters, condition)
    
    @staticmethod
    def __parse_letter(letter_element) -> Tuple[str, float]:
        return tuple((letter_element.firstChild.nodeValue,
                      float(letter_element.getAttribute("probability"))))

    def get_data_generation(self, valid_conditions: List[Condition]) -> DataGeneration:
        data_generation_element: Any = self._dom.getElementsByTagName("datageneration")[0]
        seed: int = XMLParser.__read_int_node(data_generation_element, "seed")

        conditions_element = data_generation_element.getElementsByTagName("conditions")[0]
        condition_elements = conditions_element.getElementsByTagName("condition")
        conditions: List[Tuple[Condition, int]] = [XMLParser.__parse_condition_generation(condition_element, valid_conditions) for condition_element in condition_elements]

        return DataGeneration(seed, conditions)
    
    @staticmethod
    def __parse_condition_generation(condition_element, valid_conditions: List[Condition]) -> Tuple[Condition, int]:
        condition = Condition.get_by_id(valid_conditions, condition_element.getAttribute("id"))
        return tuple((condition,
                      int(condition_element.getAttribute("samples"))))

    def get_conditions(self, valid_sequence_elements: List[SequenceElement]) -> List[Condition]:
        conditions_element: Any = self._dom.getElementsByTagName("conditions")[0]
        condition_elements = conditions_element.getElementsByTagName("condition")
        return [XMLParser.__parse_condition(condition_element, valid_sequence_elements) for condition_element in condition_elements]

    @staticmethod
    def __parse_condition(condition_element, valid_sequence_elements: List[SequenceElement]) -> Condition:
        id: str = condition_element.getAttribute("id")
        label: str = XMLParser.__read_text_node(condition_element, "label")
        description: str = XMLParser.__read_text_node(condition_element, "description")
        grammar_element: Any = condition_element.getElementsByTagName("grammar")[0]
        rule_elements = grammar_element.getElementsByTagName("rule")
        grammar: List[Rule] = [XMLParser.__parse_rule(rule_element, valid_sequence_elements) for rule_element in rule_elements]
        return Condition(id, label, description, grammar)

    @staticmethod
    def __parse_rule(rule_element, valid_sequence_elements: List[SequenceElement]) -> Rule:
        position: str = XMLParser.__read_text_node(rule_element, "position")
        
        sequence_element_elements: Any = rule_element.getElementsByTagName("sequenceelements")[0].getElementsByTagName("sequenceelement")
        sequence_elements: List[Tuple[SequenceElement, float]] = [XMLParser.__parse_sequence_element_generation(sequence_element_element, valid_sequence_elements) for sequence_element_element in sequence_element_elements]
        
        spacing_constraint_elements: Any = rule_element.getElementsByTagName("spacingconstraints")[0].getElementsByTagName("spacingconstraint")
        spacing_constraints: List[SpacingConstraint] = [XMLParser.__parse_spacing_constraint(spacing_constraint_element, valid_sequence_elements) for spacing_constraint_element in spacing_constraint_elements]

        return Rule(position, sequence_elements, spacing_constraints)
    
    @staticmethod
    def __parse_sequence_element_generation(sequence_element_element: Any, valid_sequence_elements: List[SequenceElement]) -> Tuple[SequenceElement, float]:
        sequence_element = SequenceElement.get_by_id(valid_sequence_elements, sequence_element_element.getAttribute("id"))
        return tuple((sequence_element,
                      float(sequence_element_element.getAttribute("probability"))))
    
    @staticmethod
    def __parse_spacing_constraint(spacing_constraint_element: Any, valid_sequence_elements: List[SequenceElement]) -> SpacingConstraint:
        sequence_element1: SequenceElement = SequenceElement.get_by_id(valid_sequence_elements, spacing_constraint_element.getAttribute("id1"))
        sequence_element2: SequenceElement = SequenceElement.get_by_id(valid_sequence_elements, spacing_constraint_element.getAttribute("id2"))
        min_distance: int = int(spacing_constraint_element.getAttribute("mindistance"))
        max_distance: int = int(spacing_constraint_element.getAttribute("maxdistance"))
        direction: str = spacing_constraint_element.getAttribute("direction")
        return SpacingConstraint(sequence_element1, sequence_element2, min_distance, max_distance, direction)

    def get_sequence_elements(self) -> List[SequenceElement]:
        sequence_elements_element: Any = self._dom.getElementsByTagName("sequenceelements")[0]
        id: str = sequence_elements_element.getAttribute("id")

        kmer_based_element = sequence_elements_element.getElementsByTagName("kmerbased")
        matrix_based_element: Any = sequence_elements_element.getElementsByTagName("matrixbased")
        if len(kmer_based_element) == 1:
            kmer_elements: Any = kmer_based_element[0].getElementsByTagName("kmer")
            kmers: List[Tuple[str, float]] = [XMLParser.__parse_letter(kmer_element) for kmer_element in kmer_elements]
            return KmerBasedSequenceElement(id, kmers)
        elif len(matrix_based_element) == 1:
            position_elements: Any = matrix_based_element[0].getElementsByTagName("position")
            positions: List[List[Tuple[str, float]]] = [XMLParser.__parse_position(position_element) for position_element in position_elements]
            return MatrixBasedSequenceElement(id, positions)
        else:
            raise Exception("sequence element is invalid")

    @staticmethod
    def __parse_position(position_element) -> List[Tuple[str, float]]:
        letter_elements: Any = position_element.getElementsByTagName("letter")
        return [XMLParser.__parse_letter(letter_element) for letter_element in letter_elements]
