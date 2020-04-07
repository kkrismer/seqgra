"""
MIT - CSAIL - Gifford Lab - seqgra

Implementation of Parser for XML configuration files 
(using Strategy design pattern)

@author: Konstantin Krismer
"""
import io
import logging
from typing import Any, List, Tuple
from abc import ABC, abstractmethod
from xml.dom.minidom import Document, parseString

import pkg_resources
from lxml import etree

from seqgra.parser.xmlhelper import XMLHelper
from seqgra.parser.dataparser import DataParser
from seqgra.model.data.background import Background
from seqgra.model.data.datageneration import DataGeneration, ExampleSet, Example
from seqgra.model.data.condition import Condition
from seqgra.model.data.sequenceelement import SequenceElement, KmerBasedSequenceElement, MatrixBasedSequenceElement
from seqgra.model.data.alphabetdistribution import AlphabetDistribution
from seqgra.model.data.rule import Rule
from seqgra.model.data.spacingconstraint import SpacingConstraint

class XMLDataParser(DataParser):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    def __init__(self, config: str) -> None:
        self._dom: Document = parseString(config)
        self._general_element: Any = self._dom.getElementsByTagName("general")[0]
        self.validate(config)

    def validate(self, xml_config: str) -> None:
        xsd_path = pkg_resources.resource_filename("seqgra", "data-config.xsd")
        xmlschema_doc = etree.parse(xsd_path)
        xmlschema = etree.XMLSchema(xmlschema_doc)
        xml_doc = etree.parse(io.BytesIO(xml_config.encode()))
        xmlschema.assertValid(xml_doc)
        logging.info("seqgra data configuration XML "
                     "file is well-formed and valid")

    def get_id(self) -> str:
        return self._general_element.getAttribute("id")
    
    def get_name(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "name")
    
    def get_description(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "description")
    
    def get_sequence_space(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "sequencespace")
    
    def get_type(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "type")

    def get_background(self, valid_conditions: List[Condition]) -> Background:
        background_element: Any = self._dom.getElementsByTagName("background")[0]
        min_length: int = XMLHelper.read_int_node(background_element, "minlength")
        max_length: int = XMLHelper.read_int_node(background_element, "maxlength")

        distribution_elements: Any = background_element.getElementsByTagName("alphabetdistribution")
        distributions: List[AlphabetDistribution] = \
            [XMLDataParser.__parse_alphabet_distribution(distribution_element, valid_conditions) 
             for distribution_element in distribution_elements]
        return Background(min_length, max_length, distributions)
    
    @staticmethod
    def __parse_alphabet_distribution(
        alphabet_distribution_element,
        valid_conditions: List[Condition]) -> AlphabetDistribution:
        if alphabet_distribution_element.hasAttribute("cid"):
            condition: Condition = Condition.get_by_id(valid_conditions, alphabet_distribution_element.getAttribute("cid"))
        else:
            condition: Condition = None

        if alphabet_distribution_element.hasAttribute("setname"):
            set_name: str = alphabet_distribution_element.getAttribute("setname")
        else:
            set_name: str = None

        letter_elements: Any = alphabet_distribution_element.getElementsByTagName("letter")
        letters: List[Tuple[str, float]] = \
            [XMLDataParser.__parse_letter(letter_element) 
             for letter_element in letter_elements]
        return AlphabetDistribution(letters, condition, set_name)
    
    @staticmethod
    def __parse_letter(letter_element) -> Tuple[str, float]:
        return tuple((XMLHelper.read_immediate_text_node(letter_element),
                      float(letter_element.getAttribute("probability"))))

    def get_data_generation(self, valid_conditions: List[Condition]) -> DataGeneration:
        data_generation_element: Any = self._dom.getElementsByTagName("datageneration")[0]
        seed: int = XMLHelper.read_int_node(data_generation_element, "seed")
        postprocessing_element: Any = data_generation_element.getElementsByTagName("postprocessing")
        if len(postprocessing_element) == 1:
            postprocessing_element = postprocessing_element[0]
            operation_elements = postprocessing_element.getElementsByTagName("operation")
            postprocessing: List[Tuple[str, str]] = \
                [XMLDataParser.__parse_operation(operation_element)
                 for operation_element in operation_elements]
        else:
            postprocessing: List[Tuple[str, str]] = None

        sets_element = data_generation_element.getElementsByTagName("sets")[0]
        set_elements = sets_element.getElementsByTagName("set")
        sets: List[ExampleSet] = \
            [XMLDataParser.__parse_set(set_element, valid_conditions) 
             for set_element in set_elements]
        return DataGeneration(seed, sets, postprocessing)

    @staticmethod
    def __parse_operation(operation_element) -> Tuple[str, str]:
        return tuple((XMLHelper.read_immediate_text_node(operation_element),
                      operation_element.getAttribute("labels")))
    
    @staticmethod
    def __parse_set(set_element, valid_conditions: List[Condition]) -> ExampleSet:
        name: str = set_element.getAttribute("name")
        example_elements: Any = set_element.getElementsByTagName("example")
        examples: List[Example] = \
            [XMLDataParser.__parse_example(example_element, valid_conditions)
             for example_element in example_elements]
        return ExampleSet(name, examples)
    
    @staticmethod
    def __parse_example(example_element, valid_conditions: List[Condition]) -> Example:
        samples: int = int(example_element.getAttribute("samples"))
        condition_elements: Any = example_element.getElementsByTagName("conditionref")
        conditions: List[Condition] = [Condition.get_by_id(valid_conditions, condition_element.getAttribute("cid")) 
                                       for condition_element in condition_elements]
        return Example(samples, conditions)

    def get_conditions(self, valid_sequence_elements: List[SequenceElement]) -> List[Condition]:
        conditions_element: Any = self._dom.getElementsByTagName("conditions")[0]
        condition_elements = conditions_element.getElementsByTagName("condition")
        return [XMLDataParser.__parse_condition(condition_element, valid_sequence_elements) 
                for condition_element in condition_elements]

    @staticmethod
    def __parse_condition(condition_element, valid_sequence_elements: List[SequenceElement]) -> Condition:
        id: str = condition_element.getAttribute("id")
        label: str = XMLHelper.read_text_node(condition_element, "label")
        description: str = XMLHelper.read_text_node(condition_element, "description")
        grammar_element: Any = condition_element.getElementsByTagName("grammar")[0]
        rule_elements = grammar_element.getElementsByTagName("rule")
        grammar: List[Rule] = [XMLDataParser.__parse_rule(rule_element, valid_sequence_elements)
                               for rule_element in rule_elements]
        return Condition(id, label, description, grammar)

    @staticmethod
    def __parse_rule(rule_element, valid_sequence_elements: List[SequenceElement]) -> Rule:
        position: str = XMLHelper.read_text_node(rule_element, "position")
        probability: float = XMLHelper.read_float_node(rule_element, "probability")
        
        sref_elements: Any = rule_element.getElementsByTagName("sequenceelementrefs")[0].getElementsByTagName("sequenceelementref")
        sequence_elements: List[SequenceElement] = \
            [SequenceElement.get_by_id(valid_sequence_elements, sref_element.getAttribute("sid"))
             for sref_element in sref_elements]
        
        if len(rule_element.getElementsByTagName("spacingconstraints")) == 1:
            spacing_constraint_elements: Any = rule_element.getElementsByTagName("spacingconstraints")[0].getElementsByTagName("spacingconstraint")
            spacing_constraints: List[SpacingConstraint] = \
                [XMLDataParser.__parse_spacing_constraint(spacing_constraint_element, valid_sequence_elements)
                 for spacing_constraint_element in spacing_constraint_elements]
        else:
            spacing_constraints: List[SpacingConstraint] = None

        return Rule(position, probability, sequence_elements, spacing_constraints)
    
    @staticmethod
    def __parse_spacing_constraint(spacing_constraint_element: Any, valid_sequence_elements: List[SequenceElement]) -> SpacingConstraint:
        sequence_element1: SequenceElement = SequenceElement.get_by_id(valid_sequence_elements, spacing_constraint_element.getAttribute("sid1"))
        sequence_element2: SequenceElement = SequenceElement.get_by_id(valid_sequence_elements, spacing_constraint_element.getAttribute("sid2"))
        min_distance: int = int(spacing_constraint_element.getAttribute("mindistance"))
        max_distance: int = int(spacing_constraint_element.getAttribute("maxdistance"))
        direction: str = spacing_constraint_element.getAttribute("direction")
        return SpacingConstraint(sequence_element1, sequence_element2, min_distance, max_distance, direction)

    def get_sequence_elements(self) -> List[SequenceElement]:
        sequence_elements_element: Any = self._dom.getElementsByTagName("sequenceelements")[0]
        sequence_element_elements: List[Any] = sequence_elements_element.getElementsByTagName("sequenceelement")
        return [XMLDataParser.__parse_sequence_element(sequence_element_element) 
                for sequence_element_element in sequence_element_elements]

    @staticmethod
    def __parse_sequence_element(sequence_element_element: Any) -> SequenceElement:
        id: str = sequence_element_element.getAttribute("id")
        kmer_based_element: Any = sequence_element_element.getElementsByTagName("kmerbased")
        matrix_based_element: Any = sequence_element_element.getElementsByTagName("matrixbased")
        if len(kmer_based_element) == 1:
            kmer_elements: Any = kmer_based_element[0].getElementsByTagName("kmer")
            kmers: List[Tuple[str, float]] = \
                [XMLDataParser.__parse_letter(kmer_element)
                 for kmer_element in kmer_elements]
            return KmerBasedSequenceElement(id, kmers)
        elif len(matrix_based_element) == 1:
            position_elements: Any = matrix_based_element[0].getElementsByTagName("position")
            positions: List[List[Tuple[str, float]]] = \
                [XMLDataParser.__parse_position(position_element)
                 for position_element in position_elements]
            return MatrixBasedSequenceElement(id, positions)
        else:
            raise Exception("sequence element is invalid")

    @staticmethod
    def __parse_position(position_element) -> List[Tuple[str, float]]:
        letter_elements: Any = position_element.getElementsByTagName("letter")
        return [XMLDataParser.__parse_letter(letter_element) for letter_element in letter_elements]
