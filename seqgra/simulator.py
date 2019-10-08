"""
MIT - CSAIL - Gifford Lab - seqgra

Generates synthetic sequences based on grammar

@author: Konstantin Krismer
"""

import logging
import os
import random
from typing import List, Tuple, Set

import numpy as np

from seqgra.parser.parser import Parser
from seqgra.model.background import Background
from seqgra.model.datageneration import DataGeneration, ExampleSet
from seqgra.model.condition import Condition
from seqgra.model.sequenceelement import SequenceElement
from seqgra.model.spacingconstraint import SpacingConstraint
from seqgra.model.rule import Rule
from seqgra.model.example import Example
from seqgra.logic.examplegenerator import ExampleGenerator

class Simulator:
    def __init__(self, parser: Parser) -> None:
        self._parser: Parser = parser
        self.__parse_config()
        self.check_grammar()

    def __str__(self):
        str_rep = ["Grammar:\n",
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

    def simulate_data(self, output_dir: str) -> None:
        logging.info("started data simulation")
        output_dir = output_dir.strip()
        self.__set_seed()
        self.__prepare_output_dir(output_dir)

        for example_set in self.data_generation.sets:
            self.__process_set(example_set, output_dir)
            logging.info("generated " + example_set.name + " set")
    
    def __process_set(self, example_set: ExampleSet, output_dir: str) -> None:
        condition_ids: List[str] = []
        for condition in example_set.conditions:
            condition_ids += [condition[0].id] * condition[1]
        random.shuffle(condition_ids)

        with open(output_dir + "/" + example_set.name + ".txt", "w") as data_file, \
             open(output_dir + "/" + example_set.name + "-annotation.txt", "w") as annotation_file:
            data_file.write("x\ty\n")
            annotation_file.write("annotation\ty\n")
            for condition_id in condition_ids:
                condition: Condition = Condition.get_by_id(self.conditions, condition_id)
                example: Example = ExampleGenerator.generate_example(condition, self.background)
                data_file.write(example.sequence + "\t" + condition.id + "\n")
                annotation_file.write(example.annotation + "\t" + condition.id + "\n")

    def __set_seed(self) -> None:
        random.seed(self.data_generation.seed)
        np.random.seed(self.data_generation.seed)

    def __prepare_output_dir(self, output_dir: str) -> None:
        if os.path.exists(output_dir):
            if os.path.isdir(output_dir):
                if len(os.listdir(output_dir)) > 0:
                    raise Exception("output directory non-empty")
            else:
                raise Exception("output directory cannot be created (file with same name exists)")
        else:    
            os.makedirs(output_dir)

    def check_grammar(self) -> bool:
        valid: bool = True

        c1: bool = self.check_unused_conditions()
        c2: bool = self.check_unused_sequence_elements()
        c3: bool = self.check_invalid_alphabet_distributions()
        if c3:
            c4: bool = self.check_missing_alphabet_distributions()
        else:
            c4: bool = False
        c5: bool = self.check_invalid_positions()
        c6: bool = self.check_invalid_distances()
        c7: bool = self.check_invalid_sequence_elements()
        c8: bool = self.check_spacing_contraint_se_refs()

        valid = c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8
        if valid:
            logging.info("semantic analysis of grammar completed: no issues detected")
        return valid
    
    def check_unused_conditions(self) -> bool:
        valid: bool = True

        used_condition_ids: Set[str] = set()
        for example_set in self.data_generation.sets:
            for condition_sample in example_set.conditions:
                used_condition_ids.add(condition_sample[0].id)

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

    def check_invalid_alphabet_distributions(self) -> bool:
        valid: bool = True
        
        if len(self.background.alphabet_distributions) > 1:
            for alphabet_distribution in self.background.alphabet_distributions:
                if alphabet_distribution.condition_independent:
                    valid = False
                    logging.warn("invalid definition of alphabet distributions: both condition-independent and condition-dependent distributions specified")

        return valid

    def check_missing_alphabet_distributions(self) -> bool:
        valid: bool = True

        if len(self.background.alphabet_distributions) == 1 and self.background.alphabet_distributions[0].condition_independent:
            return True
        else:
            specified_alphabets: Set[str] = set()
            for alphabet in self.background.alphabet_distributions:
                specified_alphabets.add(alphabet.condition.id)
        
            for condition in self.conditions:
                if condition.id not in specified_alphabets:
                    valid = False
                    logging.warn("no alphabet definition found for condition " + condition.id + " [cid]")

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