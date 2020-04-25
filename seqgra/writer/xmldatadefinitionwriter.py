"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file writer

@author: Konstantin Krismer
"""
from typing import Dict, List, Tuple

from lxml import etree

from seqgra.model import DataDefinition
from seqgra.model.data import Background
from seqgra.model.data import AlphabetDistribution
from seqgra.model.data import DataGeneration
from seqgra.model.data import ExampleSet
from seqgra.model.data import Example
from seqgra.model.data import PostprocessingOperation
from seqgra.model.data import Condition
from seqgra.model.data import Rule
from seqgra.model.data import SequenceElement
from seqgra.model.data import SpacingConstraint
from seqgra.model.data import MatrixBasedSequenceElement
from seqgra.model.data import KmerBasedSequenceElement
from seqgra.writer import DataDefinitionWriter


class XMLDataDefinitionWriter(DataDefinitionWriter):
    @staticmethod
    def create_root_element():
        attr_qname = etree.QName("http://www.w3.org/2001/XMLSchema-instance",
                                 "noNamespaceSchemaLocation")
        ns_map = {"xsi": "http://www.w3.org/2001/XMLSchema-instance"}

        return etree.Element("seqgradata",
                             {attr_qname: "https://seqgra.mit.edu/data-config.xsd"},
                             nsmap=ns_map)

    @staticmethod
    def attach_general_element(root, data_definition: DataDefinition) -> None:
        general_element = etree.SubElement(root, "general",
                                           {"id": data_definition.grammar_id})

        name_element = etree.SubElement(general_element, "name")
        name_element.text = data_definition.name

        description_element = etree.SubElement(general_element, "description")
        description_element.text = data_definition.description

        sequence_space_element = etree.SubElement(
            general_element, "sequencespace")
        sequence_space_element.text = data_definition.sequence_space

        model_type_element = etree.SubElement(general_element, "type")
        model_type_element.text = data_definition.model_type

    @staticmethod
    def attach_alphabet_distribution(ads_element,
                                     ad: AlphabetDistribution) -> None:
        ad_element = etree.SubElement(ads_element, "alphabetdistribution")

        if not ad.condition_independent:
            ad_element.set("cid", ad.condition.condition_id)
        if not ad.set_independent:
            ad_element.set("setname", ad.set_name)

        if ad.letters is None or len(ad.letters) == 0:
            raise Exception("no letters in alphabet")
        else:
            for letter in ad.letters:
                letter_element = etree.SubElement(ad_element, "letter",
                                                  {"probability": str(letter[1])})
                letter_element.text = letter[0]

    @staticmethod
    def attach_background_element(root, background: Background) -> None:
        background_element = etree.SubElement(root, "background")

        min_length_element = etree.SubElement(background_element, "minlength")
        min_length_element.text = str(background.min_length)

        max_length_element = etree.SubElement(background_element, "maxlength")
        max_length_element.text = str(background.max_length)

        ads_element = etree.SubElement(background_element,
                                       "alphabetdistributions")

        if background.alphabet_distributions is None or \
                len(background.alphabet_distributions) == 0:
            raise Exception("no alphabet distribution specified")
        else:
            for ad in background.alphabet_distributions:
                XMLDataDefinitionWriter.attach_alphabet_distribution(
                    ads_element, ad)

    @staticmethod
    def attach_set_element(sets_element, example_set: ExampleSet) -> None:
        if example_set.name not in {"training", "validation", "test"}:
            raise Exception("invalid set name: " + example_set.name +
                            " (valid set names: training, validation, test)")
        set_element = etree.SubElement(sets_element, "set",
                                       {"name": example_set.name})
        if example_set.examples is None or \
                len(example_set.examples) == 0:
            raise Exception("no examples specified in set " + example_set.name)
        else:
            for example in example_set.examples:
                example_element = etree.SubElement(
                    set_element, "example", {"samples": str(example.samples)})
                if example.conditions is not None and len(example.conditions) > 0:
                    for condition in example.conditions:
                        condition_element = etree.SubElement(
                            example_element, "conditionref",
                            {"cid": condition.condition_id})

    @staticmethod
    def attach_data_generation_element(root,
                                       data_generation: DataGeneration) -> None:
        dg_element = etree.SubElement(root, "datageneration")
        seed_element = etree.SubElement(dg_element, "seed")
        seed_element.text = str(data_generation.seed)

        sets_element = etree.SubElement(dg_element, "sets")
        if data_generation.sets is None or \
                len(data_generation.sets) == 0:
            raise Exception("no training, validation, test sets specified")
        else:
            for example_set in data_generation.sets:
                XMLDataDefinitionWriter.attach_set_element(
                    sets_element, example_set)

        if data_generation.postprocessing_operations is not None and \
                len(data_generation.postprocessing_operations) > 0:
            postprocessing_element = etree.SubElement(
                dg_element, "postprocessing")
            for op in data_generation.postprocessing_operations:
                op_element = etree.SubElement(postprocessing_element,
                                              "operation",
                                              {"labels": op.labels})
                op_element.text = op.name
                if op.parameters is not None and len(op.parameters) > 0:
                    for param_name, param_value in op.parameters.items():
                        op_element.set(param_name, param_value)

    @staticmethod
    def attach_sc_element(scs_element,
                          spacing_constraint: SpacingConstraint) -> None:
        etree.SubElement(scs_element, "spacingconstraint",
                         {"sid1": spacing_constraint.sequence_element1.sid,
                          "sid2": spacing_constraint.sequence_element2.sid,
                          "mindistance": str(spacing_constraint.min_distance),
                          "maxdistance": str(spacing_constraint.max_distance),
                          "order": spacing_constraint.order})

    @staticmethod
    def attach_rule_element(grammar_element, rule: Rule) -> None:
        rule_element = etree.SubElement(grammar_element, "rule")

        position_element = etree.SubElement(rule_element, "position")
        position_element.text = rule.position
        probability_element = etree.SubElement(rule_element, "probability")
        probability_element.text = str(rule.probability)

        se_refs_element = etree.SubElement(rule_element, "sequenceelementrefs")
        if rule.sequence_elements is None or \
                len(rule.sequence_elements) == 0:
            raise Exception("no sequence elements specified in rule")
        else:
            for sequence_element in rule.sequence_elements:
                se_ref_element = etree.SubElement(
                    se_refs_element, "sequenceelementref",
                    {"sid": sequence_element.sid})

        if rule.spacing_constraints is not None and \
                len(rule.spacing_constraints) > 0:
            scs_element = etree.SubElement(rule_element, "spacingconstraints")
            for spacing_constraint in rule.spacing_constraints:
                XMLDataDefinitionWriter.attach_sc_element(
                    scs_element, spacing_constraint)

    @staticmethod
    def attach_condition_element(conditions_element,
                                 condition: Condition) -> None:
        condition_element = etree.SubElement(conditions_element, "condition",
                                             {"id": condition.condition_id})

        label_element = etree.SubElement(condition_element, "label")
        label_element.text = condition.label
        description_element = etree.SubElement(
            condition_element, "description")
        description_element.text = condition.description

        grammar_element = etree.SubElement(condition_element, "grammar")

        if condition.grammar is None or \
                len(condition.grammar) == 0:
            raise Exception("no grammar specified for condition " +
                            condition.condition_id)
        else:
            for rule in condition.grammar:
                XMLDataDefinitionWriter.attach_rule_element(
                    grammar_element, rule)

    @staticmethod
    def attach_conditions_element(root,
                                  conditions: List[Condition]) -> None:
        conditions_element = etree.SubElement(root, "conditions")
        if conditions is None or len(conditions) == 0:
            raise Exception("no conditions specified")
        else:
            for condition in conditions:
                XMLDataDefinitionWriter.attach_condition_element(
                    conditions_element, condition)

    @staticmethod
    def attach_matrix_position_element(matrix_element,
                                       letters: List[Tuple[str, float]]) -> None:
        position_element = etree.SubElement(matrix_element, "position")
        if letters is None or len(letters) == 0:
            raise Exception("no letters specified in position weight matrix")
        else:
            for letter in letters:
                letter_element = etree.SubElement(position_element, "letter",
                                                  {"probability": str(letter[1])})
                letter_element.text = letter[0]

    @staticmethod
    def attach_se_element(ses_element,
                          sequence_element: SequenceElement) -> None:
        se_element = etree.SubElement(ses_element, "sequenceelement",
                                      {"id": sequence_element.sid})
        if isinstance(sequence_element, MatrixBasedSequenceElement):
            matrix_element = etree.SubElement(se_element, "matrixbased")
            for position in sequence_element.positions:
                XMLDataDefinitionWriter.attach_matrix_position_element(
                    matrix_element, position)
        elif isinstance(sequence_element, KmerBasedSequenceElement):
            kmers_element = etree.SubElement(se_element, "kmerbased")
            for kmer in sequence_element.kmers:
                kmer_element = etree.SubElement(kmers_element, "kmer",
                                                {"probability": str(kmer[1])})
                kmer_element.text = kmer[0]

    @staticmethod
    def attach_ses_element(root,
                           sequence_elements: List[SequenceElement]) -> None:
        ses_element = etree.SubElement(root, "sequenceelements")
        if sequence_elements is None or len(sequence_elements) == 0:
            raise Exception("no sequence elements specified")
        else:
            for sequence_element in sequence_elements:
                XMLDataDefinitionWriter.attach_se_element(
                    ses_element, sequence_element)

    @staticmethod
    def write_data_definition_to_file(data_definition: DataDefinition,
                                      file_name: str):
        root = XMLDataDefinitionWriter.create_root_element()
        XMLDataDefinitionWriter.attach_general_element(root, data_definition)
        XMLDataDefinitionWriter.attach_background_element(
            root, data_definition.background)
        XMLDataDefinitionWriter.attach_data_generation_element(
            root, data_definition.data_generation)
        XMLDataDefinitionWriter.attach_conditions_element(
            root, data_definition.conditions)
        XMLDataDefinitionWriter.attach_ses_element(
            root, data_definition.sequence_elements)

        with open(file_name, "wb") as config_file:
            config_file.write(etree.tostring(root, pretty_print=True,
                                             encoding="UTF-8", method="xml",
                                             xml_declaration=True))
