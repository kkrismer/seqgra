"""
MIT - CSAIL - Gifford Lab - seqgra

Implementation of Parser for XML configuration files 
(using Strategy design pattern)

@author: Konstantin Krismer
"""
import io
import logging
from typing import Any, List, Tuple, Dict
from xml.dom.minidom import Document, parseString

import pkg_resources
from lxml import etree

from seqgra import ProbabilisticToken
from seqgra.parser import XMLHelper
from seqgra.parser import DataDefinitionParser
from seqgra.model import DataDefinition
from seqgra.model.data import Background
from seqgra.model.data import DataGeneration
from seqgra.model.data import DataGenerationExample
from seqgra.model.data import DataGenerationSet
from seqgra.model.data import Condition
from seqgra.model.data import SequenceElement
from seqgra.model.data import KmerBasedSequenceElement
from seqgra.model.data import MatrixBasedSequenceElement
from seqgra.model.data import AlphabetDistribution
from seqgra.model.data import Rule
from seqgra.model.data import SpacingConstraint
from seqgra.model.data import PostprocessingOperation


class XMLDataDefinitionParser(DataDefinitionParser):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    def __init__(self, config: str) -> None:
        self.logger = logging.getLogger(__name__)
        self._dom: Document = parseString(config)
        self._general_element: Any = \
            self._dom.getElementsByTagName("general")[0]
        self.validate(config)

    def validate(self, xml_config: str) -> None:
        xsd_path = pkg_resources.resource_filename("seqgra", "data-config.xsd")
        xmlschema_doc = etree.parse(xsd_path)
        xmlschema = etree.XMLSchema(xmlschema_doc)
        xml_doc = etree.parse(io.BytesIO(xml_config.encode()))
        xmlschema.assertValid(xml_doc)
        self.logger.info("seqgra data configuration XML "
                         "file is well-formed and valid")

    def get_grammar_id(self) -> str:
        return self._general_element.getAttribute("id")

    def get_name(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "name")

    def get_description(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "description")

    def get_task(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "task")

    def get_sequence_space(self) -> str:
        return XMLHelper.read_text_node(self._general_element, "sequencespace")

    def get_seed(self) -> int:
        return XMLHelper.read_int_node(self._general_element, "seed")

    def get_background(self, valid_conditions: List[Condition]) -> Background:
        background_element: Any = \
            self._dom.getElementsByTagName("background")[0]
        min_length: int = XMLHelper.read_int_node(
            background_element, "minlength")
        max_length: int = XMLHelper.read_int_node(
            background_element, "maxlength")

        distribution_elements: Any = \
            background_element.getElementsByTagName("alphabetdistribution")
        distributions: List[AlphabetDistribution] = \
            [XMLDataDefinitionParser.__parse_alphabet_distribution(
                distribution_element, valid_conditions)
             for distribution_element in distribution_elements]
        return Background(min_length, max_length, distributions)

    @staticmethod
    def __parse_alphabet_distribution(
            alphabet_distribution_element,
            valid_conditions: List[Condition]) -> AlphabetDistribution:
        if alphabet_distribution_element.hasAttribute("cid"):
            condition: Condition = Condition.get_by_id(
                valid_conditions,
                alphabet_distribution_element.getAttribute("cid"))
        else:
            condition: Condition = None

        if alphabet_distribution_element.hasAttribute("setname"):
            set_name: str = \
                alphabet_distribution_element.getAttribute("setname")
        else:
            set_name: str = None

        letter_elements: Any = \
            alphabet_distribution_element.getElementsByTagName("letter")
        letters: List[ProbabilisticToken] = \
            [XMLDataDefinitionParser.__parse_letter(letter_element)
             for letter_element in letter_elements]
        return AlphabetDistribution(letters, condition, set_name)

    @staticmethod
    def __parse_letter(letter_element) -> ProbabilisticToken:
        return ProbabilisticToken(
            XMLHelper.read_immediate_text_node(letter_element),
            float(letter_element.getAttribute("probability")))

    def get_data_generation(
            self,
            valid_conditions: List[Condition]) -> DataGeneration:
        data_generation_element: Any = \
            self._dom.getElementsByTagName("datageneration")[0]
        postprocessing_element: Any = \
            data_generation_element.getElementsByTagName("postprocessing")
        if len(postprocessing_element) == 1:
            postprocessing_element = postprocessing_element[0]
            operation_elements = \
                postprocessing_element.getElementsByTagName("operation")
            postprocessing: List[PostprocessingOperation] = \
                [XMLDataDefinitionParser.__parse_operation(operation_element)
                 for operation_element in operation_elements]
        else:
            postprocessing: List[Tuple[str, str]] = None

        sets_element = data_generation_element.getElementsByTagName("sets")[0]
        set_elements = sets_element.getElementsByTagName("set")
        sets: List[DataGenerationSet] = \
            [XMLDataDefinitionParser.__parse_set(set_element, valid_conditions)
             for set_element in set_elements]
        return DataGeneration(sets, postprocessing)

    @staticmethod
    def __parse_operation(operation_element) -> Tuple[str, str]:
        name: str = XMLHelper.read_immediate_text_node(operation_element)
        labels: str = operation_element.getAttribute("labels")
        all_params: List[Tuple[str, str]] = \
            operation_element.attributes.items()
        if len(all_params) > 1:
            # remove labels attribute
            all_params.pop(0)
            parameters: Dict[str, str] = dict(all_params)
        else:
            parameters: Dict[str, str] = None

        return PostprocessingOperation(name, labels, parameters)

    @staticmethod
    def __parse_set(set_element,
                    valid_conditions: List[Condition]) -> DataGenerationSet:
        name: str = set_element.getAttribute("name")
        example_elements: Any = set_element.getElementsByTagName("example")
        examples: List[DataGenerationExample] = \
            [XMLDataDefinitionParser.__parse_example(
                example_element, valid_conditions)
             for example_element in example_elements]
        return DataGenerationSet(name, examples)

    @staticmethod
    def __parse_example(
            example_element,
            valid_conditions: List[Condition]) -> DataGenerationExample:
        samples: int = int(example_element.getAttribute("samples"))
        condition_elements: Any = \
            example_element.getElementsByTagName("conditionref")
        conditions: List[Condition] = \
            [Condition.get_by_id(valid_conditions,
                                 condition_element.getAttribute("cid"))
             for condition_element in condition_elements]
        return DataGenerationExample(samples, conditions)

    def get_conditions(
            self,
            valid_sequence_elements: List[SequenceElement]) -> List[Condition]:
        conditions_element: Any = \
            self._dom.getElementsByTagName("conditions")[0]
        condition_elements: Any = \
            conditions_element.getElementsByTagName("condition")
        return [XMLDataDefinitionParser.__parse_condition(
            condition_element, valid_sequence_elements)
            for condition_element in condition_elements]

    @staticmethod
    def __parse_condition(
            condition_element,
            valid_sequence_elements: List[SequenceElement]) -> Condition:
        cid: str = condition_element.getAttribute("id")
        label: str = XMLHelper.read_text_node(condition_element, "label")
        description: str = XMLHelper.read_text_node(
            condition_element, "description")
        grammar_element: Any = \
            condition_element.getElementsByTagName("grammar")[0]
        rule_elements = grammar_element.getElementsByTagName("rule")
        grammar: List[Rule] = \
            [XMLDataDefinitionParser.__parse_rule(
                rule_element, valid_sequence_elements)
             for rule_element in rule_elements]
        return Condition(cid, label, description, grammar)

    @staticmethod
    def __parse_rule(rule_element,
                     valid_sequence_elements: List[SequenceElement]) -> Rule:
        position: str = XMLHelper.read_text_node(rule_element, "position")
        probability: float = XMLHelper.read_float_node(
            rule_element, "probability")

        sref_elements: Any = rule_element.getElementsByTagName(
            "sequenceelementrefs")[0].getElementsByTagName("sequenceelementref")
        sequence_elements: List[SequenceElement] = \
            [SequenceElement.get_by_id(valid_sequence_elements,
                                       sref_element.getAttribute("sid"))
             for sref_element in sref_elements]

        if len(rule_element.getElementsByTagName("spacingconstraints")) == 1:
            spacing_constraint_elements: Any = rule_element.getElementsByTagName(
                "spacingconstraints")[0].getElementsByTagName("spacingconstraint")
            spacing_constraints: List[SpacingConstraint] = \
                [XMLDataDefinitionParser.__parse_spacing_constraint(
                    spacing_constraint_element, valid_sequence_elements)
                 for spacing_constraint_element in spacing_constraint_elements]
        else:
            spacing_constraints: List[SpacingConstraint] = None

        return Rule(position, probability, sequence_elements,
                    spacing_constraints)

    @staticmethod
    def __parse_spacing_constraint(
            spacing_constraint_element: Any,
            valid_sequence_elements: List[SequenceElement]) -> SpacingConstraint:
        sequence_element1: SequenceElement = SequenceElement.get_by_id(
            valid_sequence_elements,
            spacing_constraint_element.getAttribute("sid1"))
        sequence_element2: SequenceElement = SequenceElement.get_by_id(
            valid_sequence_elements,
            spacing_constraint_element.getAttribute("sid2"))
        min_distance: int = int(
            spacing_constraint_element.getAttribute("mindistance"))
        max_distance: int = int(
            spacing_constraint_element.getAttribute("maxdistance"))
        order: str = spacing_constraint_element.getAttribute("order")
        return SpacingConstraint(sequence_element1, sequence_element2,
                                 min_distance, max_distance, order)

    def get_sequence_elements(self) -> List[SequenceElement]:
        sequence_elements_element: Any = \
            self._dom.getElementsByTagName("sequenceelements")[0]
        sequence_element_elements: List[Any] = \
            sequence_elements_element.getElementsByTagName("sequenceelement")
        return [XMLDataDefinitionParser.__parse_sequence_element(sequence_element_element)
                for sequence_element_element in sequence_element_elements]

    @staticmethod
    def __parse_sequence_element(sequence_element_element: Any) -> SequenceElement:
        sid: str = sequence_element_element.getAttribute("id")
        kmer_based_element: Any = \
            sequence_element_element.getElementsByTagName("kmerbased")
        matrix_based_element: Any = \
            sequence_element_element.getElementsByTagName("matrixbased")
        if len(kmer_based_element) == 1:
            kmer_elements: Any = \
                kmer_based_element[0].getElementsByTagName("kmer")
            kmers: List[ProbabilisticToken] = \
                [XMLDataDefinitionParser.__parse_letter(kmer_element)
                 for kmer_element in kmer_elements]
            return KmerBasedSequenceElement(sid, kmers)
        elif len(matrix_based_element) == 1:
            position_elements: Any = \
                matrix_based_element[0].getElementsByTagName("position")
            positions: List[List[ProbabilisticToken]] = \
                [XMLDataDefinitionParser.__parse_position(position_element)
                 for position_element in position_elements]
            return MatrixBasedSequenceElement(sid, positions)
        else:
            raise Exception("sequence element is invalid")

    @staticmethod
    def __parse_position(position_element) -> List[ProbabilisticToken]:
        letter_elements: Any = position_element.getElementsByTagName("letter")
        return [XMLDataDefinitionParser.__parse_letter(letter_element)
                for letter_element in letter_elements]

    def get_data_definition(self) -> DataDefinition:
        sequence_elements: List[SequenceElement] = self.get_sequence_elements()
        conditions: List[Condition] = self.get_conditions(sequence_elements)
        return DataDefinition(self.get_grammar_id(),
                              self.get_name(),
                              self.get_description(),
                              self.get_task(),
                              self.get_sequence_space(),
                              self.get_seed(),
                              self.get_background(conditions),
                              self.get_data_generation(conditions),
                              conditions,
                              sequence_elements)
