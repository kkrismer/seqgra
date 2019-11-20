"""
MIT - CSAIL - Gifford Lab - seqgra

Generates synthetic sequences based on grammar

@author: Konstantin Krismer
"""
import logging
import os
import sys
import random
from typing import List, Tuple, Set, Dict

import numpy as np
import pkg_resources

from seqgra.parser.dataparser import DataParser
from seqgra.model.data.background import Background
from seqgra.model.data.datageneration import DataGeneration, ExampleSet
from seqgra.model.data.condition import Condition
from seqgra.model.data.sequenceelement import SequenceElement
from seqgra.model.data.spacingconstraint import SpacingConstraint
from seqgra.model.data.rule import Rule
from seqgra.model.data.example import Example
from seqgra.logic.examplegenerator import ExampleGenerator

class Simulator:
    def __init__(self, parser: DataParser, output_dir: str) -> None:
        self._parser: DataParser = parser
        self.__parse_config()
        self.check_grammar()
        output_dir = output_dir.strip().replace("\\", "/")
        if not output_dir.endswith("/"):
            output_dir += "/"
        self.output_dir = output_dir + self.id + "/"
        self.__prepare_output_dir()

    def __str__(self):
        str_rep = ["seqgra data configuration:\n",
        "\tID: ", self.id, "\n",
        "\tLabel: ", self.label, "\n",
        "\tDescription:\n"]
        if self.description:
            str_rep += ["\t", self.description, "\n"]
        str_rep += ["\tSequence space: ", self.sequence_space, "\n"]

        str_rep += ["\t" + s + "\n" for s in str(self.background).splitlines()]
        str_rep += ["\t" + s + "\n" for s in str(self.data_generation).splitlines()]

        str_rep += ["\tConditions:\n"]
        for condition in self.conditions:
            str_rep += ["\t\t" + s + "\n" for s in str(condition).splitlines()]
        
        str_rep += ["\tSequence elements:\n"]
        for sequence_element in self.sequence_elements:
            str_rep += ["\t\t" + s + "\n" for s in str(sequence_element).splitlines()]
        return "".join(str_rep)

    def __parse_config(self):
        self.id: str = self._parser.get_id()
        self.label: str = self._parser.get_label()
        self.description: str = self._parser.get_description()
        self.sequence_space: str = self._parser.get_sequence_space()
        self.sequence_elements: List[SequenceElement] = self._parser.get_sequence_elements()
        self.conditions: List[Condition] = self._parser.get_conditions(self.sequence_elements)
        self.background: Background = self._parser.get_background(self.conditions)
        self.data_generation: DataGeneration = self._parser.get_data_generation(self.conditions)

    def simulate_data(self) -> None:
        logging.info("started data simulation")
        
        if len(os.listdir(self.output_dir)) > 0:
            raise Exception("output directory non-empty")

        # write session info to file
        self.write_session_info()

        self.__set_seed()

        for example_set in self.data_generation.sets:
            self.__process_set(example_set)
            logging.info("generated " + example_set.name + " set")

    def write_session_info(self) -> None:
        with open(self.output_dir + "session-info.txt", "w") as session_file:
            session_file.write("seqgra package version: " + pkg_resources.require("seqgra")[0].version + "\n")
            session_file.write("NumPy version: " + np.version.version + "\n")
            session_file.write("Python version: " + sys.version + "\n")

    def __process_set(self, example_set: ExampleSet) -> None:
        condition_ids: List[str] = []
        for example in example_set.examples:
            condition_ids += [self.__serialize_example(example)] * example.samples
        random.shuffle(condition_ids)

        with open(self.output_dir + "/" + example_set.name + ".txt", "w") as data_file, \
             open(self.output_dir + "/" + example_set.name + "-annotation.txt", "w") as annotation_file:
            data_file.write("x\ty\n")
            annotation_file.write("annotation\ty\n")
            for condition_id in condition_ids:
                conditions: List[Condition] = self.__deserialize_example(condition_id)
                example: Example = ExampleGenerator.generate_example(conditions, example_set.name, self.background)
                data_file.write(example.sequence + "\t" + condition_id + "\n")
                annotation_file.write(example.annotation + "\t" + condition_id + "\n")

    def __serialize_example(self, example: Example) -> str:
        return "|".join([condition.id for condition in example.conditions])

    def __deserialize_example(self, example_str_rep: str) -> List[Condition]:
        condition_ids: List[str] = example_str_rep.split("|")
        return [Condition.get_by_id(self.conditions, condition_id) for condition_id in condition_ids]

    def __set_seed(self) -> None:
        random.seed(self.data_generation.seed)
        np.random.seed(self.data_generation.seed)

    def __prepare_output_dir(self) -> None:
        if os.path.exists(self.output_dir):
            if not os.path.isdir(self.output_dir):
                raise Exception("output directory cannot be created (file with same name exists)")
        else:    
            os.makedirs(self.output_dir)

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
            logging.info("semantic analysis of grammar completed: no issues detected")
        return valid
    
    def check_unused_conditions(self) -> bool:
        valid: bool = True

        used_condition_ids: Set[str] = set()
        for example_set in self.data_generation.sets:
            for example in example_set.examples:
                for condition_sample in example.conditions:
                    used_condition_ids.add(condition_sample.id)

        for condition in self.conditions:
            if condition.id not in used_condition_ids:
                valid = False
                logging.warn("condition " + condition.id + " [cid]: unused condition")

        return valid

    def check_unused_sequence_elements(self) -> bool:
        valid: bool = True

        used_sequence_element_ids: Set[str] = set()
        for condition in self.conditions:
            for rule in condition.grammar:
                for sequence_element in rule.sequence_elements:
                    used_sequence_element_ids.add(sequence_element.id)

        for sequence_element in self.sequence_elements:
            if sequence_element.id not in used_sequence_element_ids:
                valid = False
                logging.warn("sequence element " + sequence_element.id + " [sid]: unused sequence element")
        
        return valid

    def check_missing_alphabet_distributions(self) -> bool:
        valid: bool = True

        set_condition_combinations: Dict[str, Dict[str, str]] = dict()

        for example_set in self.data_generation.sets:
            for example in example_set.examples:
                for condition_sample in example.conditions:
                    if example_set.name in set_condition_combinations:
                        set_condition_combinations[example_set.name][condition_sample.id] = "unspecified"
                    else:
                        set_condition_combinations[example_set.name] = {condition_sample.id: "unspecified"}

        for alphabet in self.background.alphabet_distributions:
            if alphabet.set_independent and alphabet.condition_independent:
                for set_name in set_condition_combinations.keys():
                    tmp_dict = set_condition_combinations[set_name]
                    for condition_id in tmp_dict.keys():
                        if set_condition_combinations[set_name][condition_id] == "global":
                            valid = False
                            logging.warn("more than one global alphabet definition found")
                        else:
                            set_condition_combinations[set_name][condition_id] = "global"
            elif alphabet.condition_independent:
                tmp_dict = set_condition_combinations[alphabet.set_name]
                for condition_id in tmp_dict.keys():
                    if tmp_dict[condition_id] == "condition-independent":
                        valid = False
                        logging.warn("more than one condition-independent alphabet definition found for set " + alphabet.set_name)
                    else:
                        tmp_dict[condition_id] = "condition-independent"
            elif alphabet.set_independent:
                for set_name in set_condition_combinations.keys():
                    if set_condition_combinations[set_name][alphabet.condition.id] == "set-independent":
                        valid = False
                        logging.warn("more than one set-independent alphabet definition found for condition " + alphabet.condition.id + " [cid]")
                    else:
                        set_condition_combinations[set_name][alphabet.condition.id] = "set-independent"
            else:
                if set_condition_combinations[alphabet.set_name][alphabet.condition.id] == "specified":
                    valid = False
                    logging.warn("duplicate alphabet definition found for set name " + example_set.name + " and condition " + condition_sample.id + " [cid]")
                else:
                    set_condition_combinations[alphabet.set_name][alphabet.condition.id] = "specified"

        for set_name, tmp_dict in set_condition_combinations.items():
            for condition_id, value in tmp_dict.items():
                if value == "unspecified":
                    valid = False
                    logging.warn("no alphabet definition found for set name " + set_name + " and condition " + condition_id + " [cid]")

        return valid

    def check_invalid_positions(self) -> bool:
        valid: bool = True
        for condition in self.conditions:
            for i in range(len(condition.grammar)):
                rule = condition.grammar[i]
                if rule.position != "random" and rule.position != "start" and rule.position != "end" and rule.position != "center":
                    if int(rule.position) > self.background.min_length:
                        valid = False
                        logging.warn("condition " + condition.id + " [cid], rule " + str(i + 1) + ": position exceeds minimum sequence length")
                    elif int(rule.probability) + self.__get_longest_sequence_element_length(rule) > self.background.min_length:
                        valid = False
                        logging.warn("condition " + condition.id + " [cid], rule " + str(i + 1) + ": position plus sequence element length exceeds minimum sequence length")
        return valid
        
    def check_invalid_distances(self) -> bool:
        valid: bool = True

        for condition in self.conditions:
            for i in range(len(condition.grammar)):
                rule: Rule = condition.grammar[i]
                if rule.spacing_constraints is not None and len(rule.spacing_constraints) > 0:
                    for j in range(len(rule.spacing_constraints)):
                        spacing_constraint: SpacingConstraint = rule.spacing_constraints[j]
                        if spacing_constraint.min_distance > self.background.min_length:
                            valid = False
                            logging.warn("condition " + condition.id + " [cid], rule " + str(i + 1) + ", spacing constraint " + str(j + 1) + ": minimum distance exceeds minimum sequence length")
                        elif spacing_constraint.min_distance + spacing_constraint.sequence_element1.get_max_length() + spacing_constraint.sequence_element2.get_max_length() > self.background.min_length:
                            valid = False
                            logging.warn("condition " + condition.id + " [cid], rule " + str(i + 1) + ", spacing constraint " + str(j + 1) + ": minimum distance plus sequence element lengths exceeds minimum sequence length")

        return valid

    def check_invalid_sequence_elements(self) -> bool:
        valid: bool = True
        
        for sequence_element in self.sequence_elements:
            if sequence_element.get_max_length() > self.background.min_length:
                valid = False
                logging.warn("sequence element " + sequence_element.id + ": maximum sequence element length exceeds minimum sequence length")
        
        return valid

    def check_spacing_contraint_se_refs(self) -> bool:
        valid: bool = True
        valid_sequence_element_ids: Set[str] = set()
        for condition in self.conditions:
            for i in range(len(condition.grammar)):
                rule: Rule = condition.grammar[i]
                if rule.spacing_constraints is not None and len(rule.spacing_constraints) > 0:
                    valid_sequence_element_ids.clear()
                    for sequence_element in rule.sequence_elements:
                        valid_sequence_element_ids.add(sequence_element.id)

                    for j in range(len(rule.spacing_constraints)):
                        spacing_constraint: SpacingConstraint = rule.spacing_constraints[j]
                        if spacing_constraint.sequence_element1.id not in valid_sequence_element_ids:
                            valid = False
                            logging.error("condition " + condition.id + " [cid], rule " + str(i + 1) + \
                                ", spacing constraint " + str(j + 1) + ": sequence element " + \
                                spacing_constraint.sequence_element1.id + " [sid] not among sequence elements of rule")
                        if spacing_constraint.sequence_element2.id not in valid_sequence_element_ids:
                            valid = False
                            logging.error("condition " + condition.id + " [cid], rule " + str(i + 1) + \
                                ", spacing constraint " + str(j + 1) + ": sequence element " + \
                                spacing_constraint.sequence_element2.id + " [sid] not among sequence elements of rule")

        return valid

    def __get_longest_sequence_element_length(self, rule: Rule):
        longest_length: int = 0

        for sequence_element in rule.sequence_elements:
            if sequence_element.get_max_length() > longest_length:
                longest_length = sequence_element.get_max_length()
        return longest_length