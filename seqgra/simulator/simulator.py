"""
MIT - CSAIL - Gifford Lab - seqgra

Generates synthetic sequences based on grammar

@author: Konstantin Krismer
"""
from __future__ import annotations

import logging
import os
import sys
import random
from typing import List, Set, Dict

import numpy as np
import pkg_resources
import ushuffle

from seqgra import MiscHelper
from seqgra.model import DataDefinition
from seqgra.model.data import ExampleSet
from seqgra.model.data import Condition
from seqgra.model.data import SpacingConstraint
from seqgra.model.data import Rule
from seqgra.simulator import Example
from seqgra.simulator import ExampleGenerator


class Simulator:
    def __init__(self, data_definition: DataDefinition, output_dir: str) -> None:
        self.definition: DataDefinition = data_definition
        self.check_grammar()
        self.output_dir = MiscHelper.prepare_path(output_dir + "/" +
                                                  self.definition.grammar_id)

    def simulate_data(self) -> None:
        logging.info("started data simulation")

        if len(os.listdir(self.output_dir)) > 0:
            raise Exception("output directory non-empty")

        # write session info to file
        self.write_session_info()

        self.__set_seed()

        for example_set in self.definition.data_generation.sets:
            self.__process_set(example_set)
            logging.info("generated " + example_set.name + " set")

        if self.definition.data_generation.postprocessing_operations is not None:
            for example_set in self.definition.data_generation.sets:
                for operation in self.definition.data_generation.postprocessing_operations:
                    if operation.name == "kmer-frequency-preserving-shuffle":
                        if operation.parameters is not None and "k" in operation.parameters:
                            self.__add_shuffled_examples(
                                example_set.name,
                                int(operation.parameters["k"]),
                                operation.labels)
                        else:
                            self.__add_shuffled_examples(
                                example_set.name, 1, operation.labels)

    def __add_shuffled_examples(self, set_name: str,
                                preserve_frequencies_for_kmer: int,
                                labels_value: str,
                                background_character: str = "_") -> None:
        # write shuffled examples
        with open(self.output_dir + "/" + set_name + ".txt", "r") as data_file:
            with open(self.output_dir + "/" + set_name + "-shuffled.txt", "w") as shuffled_data_file:
                next(data_file)
                for line in data_file:
                    columns = line.split("\t")
                    shuffled_data_file.write(self.__shuffle_example(
                        columns[0], preserve_frequencies_for_kmer) + "\t" +
                        labels_value + "\n")

        # write annotations for shuffled examples (all background)
        with open(self.output_dir + "/" + set_name + "-annotation.txt", "r") as annotation_file:
            with open(self.output_dir + "/" + set_name + "-annotation-shuffled.txt", "w") as shuffled_annotation_file:
                next(annotation_file)
                for line in annotation_file:
                    columns = line.split("\t")
                    shuffled_annotation_file.write(
                        "".join([background_character] * len(columns[0])) +
                        "\t" + labels_value + "\n")

        # merge files
        with open(self.output_dir + "/" + set_name + ".txt", "a") as data_file:
            with open(self.output_dir + "/" + set_name + "-shuffled.txt", "r") as shuffled_data_file:
                for line in shuffled_data_file:
                    data_file.write(line)
        with open(self.output_dir + "/" + set_name + "-annotation.txt", "a") as annotation_file:
            with open(self.output_dir + "/" + set_name + "-annotation-shuffled.txt", "r") as shuffled_annotation_file:
                for line in shuffled_annotation_file:
                    annotation_file.write(line)

        # delete superfluous files
        os.remove(self.output_dir + "/" + set_name + "-shuffled.txt")
        os.remove(self.output_dir + "/" + set_name +
                  "-annotation-shuffled.txt")

    def __shuffle_example(self, example: str,
                          preserve_frequencies_for_kmer: int = 1) -> str:
        if preserve_frequencies_for_kmer > 1:
            return str(ushuffle.shuffle(example.encode(),
                                        preserve_frequencies_for_kmer), 'utf-8')
        else:
            example_list = list(example)
            random.shuffle(example_list)
            return "".join(example_list)

    def write_session_info(self) -> None:
        with open(self.output_dir + "session-info.txt", "w") as session_file:
            session_file.write("seqgra package version: " +
                               pkg_resources.require("seqgra")[0].version + "\n")
            session_file.write("NumPy version: " + np.version.version + "\n")
            session_file.write("Python version: " + sys.version + "\n")

    def __process_set(self, example_set: ExampleSet) -> None:
        condition_ids: List[str] = []
        for example in example_set.examples:
            condition_ids += [self.__serialize_example(example)] * \
                example.samples
        random.shuffle(condition_ids)

        with open(self.output_dir + "/" + example_set.name + ".txt", "w") as data_file, \
                open(self.output_dir + "/" + example_set.name + "-annotation.txt", "w") as annotation_file:
            data_file.write("x\ty\n")
            annotation_file.write("annotation\ty\n")
            for condition_id in condition_ids:
                conditions: List[Condition] = self.__deserialize_example(
                    condition_id)
                example: Example = ExampleGenerator.generate_example(
                    conditions, example_set.name, self.definition.background)
                data_file.write(example.sequence + "\t" + condition_id + "\n")
                annotation_file.write(
                    example.annotation + "\t" + condition_id + "\n")

    def __serialize_example(self, example: Example) -> str:
        return "|".join([condition.condition_id
                         for condition in example.conditions])

    def __deserialize_example(self, example_str_rep: str) -> List[Condition]:
        condition_ids: List[str] = example_str_rep.split("|")
        return [Condition.get_by_id(self.definition.conditions, condition_id)
                for condition_id in condition_ids]

    def __set_seed(self) -> None:
        random.seed(self.definition.data_generation.seed)
        np.random.seed(self.definition.data_generation.seed)
        ushuffle.set_seed(self.definition.data_generation.seed)

    def check_grammar(self) -> bool:
        valid: bool = True

        c1: bool = self.check_unused_conditions()
        c2: bool = self.check_unused_sequence_elements()
        c3: bool = self.check_missing_alphabet_distributions()
        c4: bool = self.check_invalid_positions()
        c5: bool = self.check_invalid_distances()
        c6: bool = self.check_invalid_sequence_elements()
        c7: bool = self.check_spacing_contraint_se_refs()

        valid = c1 and c2 and c3 and c4 and c5 and c6 and c7
        if valid:
            logging.info(
                "semantic analysis of grammar completed: no issues detected")
        return valid

    def check_unused_conditions(self) -> bool:
        valid: bool = True

        used_condition_ids: Set[str] = set()
        for example_set in self.definition.data_generation.sets:
            for example in example_set.examples:
                for condition_sample in example.conditions:
                    used_condition_ids.add(condition_sample.condition_id)

        for condition in self.definition.conditions:
            if condition.condition_id not in used_condition_ids:
                valid = False
                logging.warn("condition " + condition.condition_id +
                             " [cid]: unused condition")

        return valid

    def check_unused_sequence_elements(self) -> bool:
        valid: bool = True

        used_sequence_element_ids: Set[str] = set()
        for condition in self.definition.conditions:
            for rule in condition.grammar:
                for sequence_element in rule.sequence_elements:
                    used_sequence_element_ids.add(sequence_element.sid)

        for sequence_element in self.definition.sequence_elements:
            if sequence_element.sid not in used_sequence_element_ids:
                valid = False
                logging.warn("sequence element " + sequence_element.sid +
                             " [sid]: unused sequence element")

        return valid

    def check_missing_alphabet_distributions(self) -> bool:
        valid: bool = True

        set_condition_combinations: Dict[str, Dict[str, str]] = dict()

        for example_set in self.definition.data_generation.sets:
            for example in example_set.examples:
                for condition_sample in example.conditions:
                    if example_set.name in set_condition_combinations:
                        set_condition_combinations[example_set.name][condition_sample.condition_id] = "unspecified"
                    else:
                        set_condition_combinations[example_set.name] = {
                            condition_sample.condition_id: "unspecified"}

        for alphabet in self.definition.background.alphabet_distributions:
            if alphabet.set_independent and alphabet.condition_independent:
                for set_name in set_condition_combinations.keys():
                    tmp_dict = set_condition_combinations[set_name]
                    for condition_id in tmp_dict.keys():
                        if set_condition_combinations[set_name][condition_id] == "global":
                            valid = False
                            logging.warn("more than one global alphabet "
                                         "definition found")
                        else:
                            set_condition_combinations[set_name][condition_id] = "global"
            elif alphabet.condition_independent:
                tmp_dict = set_condition_combinations[alphabet.set_name]
                for condition_id in tmp_dict.keys():
                    if tmp_dict[condition_id] == "condition-independent":
                        valid = False
                        logging.warn("more than one condition-independent "
                                     "alphabet definition found for set " +
                                     alphabet.set_name)
                    else:
                        tmp_dict[condition_id] = "condition-independent"
            elif alphabet.set_independent:
                for set_name in set_condition_combinations.keys():
                    if set_condition_combinations[set_name][alphabet.condition.condition_id] == "set-independent":
                        valid = False
                        logging.warn("more than one set-independent alphabet "
                                     "definition found for condition " +
                                     alphabet.condition.condition_id + " [cid]")
                    else:
                        set_condition_combinations[set_name][alphabet.condition.condition_id] = "set-independent"
            else:
                if set_condition_combinations[alphabet.set_name][alphabet.condition.condition_id] == "specified":
                    valid = False
                    logging.warn("duplicate alphabet definition found for "
                                 "set name " + alphabet.set_name +
                                 " and condition " +
                                 alphabet.condition_id + " [cid]")
                else:
                    set_condition_combinations[alphabet.set_name][alphabet.condition.condition_id] = "specified"

        for set_name, tmp_dict in set_condition_combinations.items():
            for condition_id, value in tmp_dict.items():
                if value == "unspecified":
                    valid = False
                    logging.warn("no alphabet definition found for set name " +
                                 set_name + " and condition " +
                                 condition_id + " [cid]")

        return valid

    def check_invalid_positions(self) -> bool:
        valid: bool = True
        for condition in self.definition.conditions:
            for i in range(len(condition.grammar)):
                rule = condition.grammar[i]
                if rule.position != "random" and rule.position != "start" \
                        and rule.position != "end" and rule.position != "center":
                    if int(rule.position) > self.definition.background.min_length:
                        valid = False
                        logging.warn("condition " + condition.condition_id +
                                     " [cid], rule " + str(i + 1) +
                                     ": position exceeds minimum "
                                     "sequence length")
                    elif int(rule.probability) + self.__get_longest_sequence_element_length(rule) > self.definition.background.min_length:
                        valid = False
                        logging.warn("condition " + condition.condition_id +
                                     " [cid], rule " + str(i + 1) +
                                     ": position plus sequence element "
                                     "length exceeds minimum sequence length")
        return valid

    def check_invalid_distances(self) -> bool:
        valid: bool = True

        for condition in self.definition.conditions:
            for i in range(len(condition.grammar)):
                rule: Rule = condition.grammar[i]
                if rule.spacing_constraints is not None and \
                        len(rule.spacing_constraints) > 0:
                    for j in range(len(rule.spacing_constraints)):
                        spacing_constraint: SpacingConstraint = rule.spacing_constraints[j]
                        if spacing_constraint.min_distance > self.definition.background.min_length:
                            valid = False
                            logging.warn("condition " + condition.condition_id +
                                         " [cid], rule " + str(i + 1) +
                                         ", spacing constraint " + str(j + 1) +
                                         ": minimum distance exceeds minimum "
                                         "sequence length")
                        elif spacing_constraint.min_distance + spacing_constraint.sequence_element1.get_max_length() + spacing_constraint.sequence_element2.get_max_length() > self.definition.background.min_length:
                            valid = False
                            logging.warn("condition " + condition.condition_id +
                                         " [cid], rule " + str(i + 1) +
                                         ", spacing constraint " + str(j + 1) +
                                         ": minimum distance plus sequence "
                                         "element lengths exceeds minimum "
                                         "sequence length")

        return valid

    def check_invalid_sequence_elements(self) -> bool:
        valid: bool = True

        for sequence_element in self.definition.sequence_elements:
            if sequence_element.get_max_length() > self.definition.background.min_length:
                valid = False
                logging.warn("sequence element " + sequence_element.sid +
                             ": maximum sequence element length exceeds "
                             "minimum sequence length")

        return valid

    def check_spacing_contraint_se_refs(self) -> bool:
        valid: bool = True
        valid_sequence_element_ids: Set[str] = set()
        for condition in self.definition.conditions:
            for i in range(len(condition.grammar)):
                rule: Rule = condition.grammar[i]
                if rule.spacing_constraints is not None and len(rule.spacing_constraints) > 0:
                    valid_sequence_element_ids.clear()
                    for sequence_element in rule.sequence_elements:
                        valid_sequence_element_ids.add(sequence_element.sid)

                    for j in range(len(rule.spacing_constraints)):
                        spacing_constraint: SpacingConstraint = rule.spacing_constraints[j]
                        if spacing_constraint.sequence_element1.sid not in valid_sequence_element_ids:
                            valid = False
                            logging.error("condition " + condition.condition_id +
                                          " [cid], rule " + str(i + 1) +
                                          ", spacing constraint " +
                                          str(j + 1) + ": sequence element " +
                                          spacing_constraint.sequence_element1.sid +
                                          " [sid] not among sequence elements "
                                          "of rule")
                        if spacing_constraint.sequence_element2.sid not in valid_sequence_element_ids:
                            valid = False
                            logging.error("condition " + condition.condition_id +
                                          " [cid], rule " + str(i + 1) +
                                          ", spacing constraint " +
                                          str(j + 1) + ": sequence element " +
                                          spacing_constraint.sequence_element2.sid +
                                          " [sid] not among sequence elements "
                                          "of rule")

        return valid

    def __get_longest_sequence_element_length(self, rule: Rule):
        longest_length: int = 0

        for sequence_element in rule.sequence_elements:
            if sequence_element.get_max_length() > longest_length:
                longest_length = sequence_element.get_max_length()
        return longest_length
