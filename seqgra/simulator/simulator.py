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
import subprocess
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pkg_resources
import ushuffle

import seqgra.constants as c
from seqgra import AnnotatedExample, MiscHelper
from seqgra.model import DataDefinition
from seqgra.model.data import DataGenerationExample
from seqgra.model.data import DataGenerationSet
from seqgra.model.data import Condition
from seqgra.model.data import SpacingConstraint
from seqgra.model.data import Rule
from seqgra.simulator import ExampleGenerator


class Simulator:
    def __init__(self, data_definition: DataDefinition,
                 output_dir: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.definition: DataDefinition = data_definition
        self.check_grammar()
        self.output_dir = MiscHelper.prepare_path(output_dir + "/" +
                                                  self.definition.grammar_id)

    def simulate_data(self) -> None:
        self.logger.info("started data simulation")

        if len(os.listdir(self.output_dir)) > 0:
            raise Exception("output directory non-empty")

        # write session info to file
        self.write_session_info()

        self.__set_seed()

        for example_set in self.definition.data_generation.sets:
            self.__process_set(example_set)
            self.logger.info("generated %s set", example_set.name)

        if self.definition.data_generation.postprocessing_operations is not None:
            for example_set in self.definition.data_generation.sets:
                for operation in self.definition.data_generation.postprocessing_operations:
                    if operation.name == "kmer-frequency-preserving-shuffle":
                        if operation.parameters is not None and \
                                "k" in operation.parameters:
                            self.__add_shuffled_examples(
                                example_set.name,
                                int(operation.parameters["k"]),
                                operation.labels)
                        else:
                            self.__add_shuffled_examples(
                                example_set.name, 1, operation.labels)

    def create_grammar_heatmap(self, set_name: str) -> None:
        file_name: str = self.output_dir + "/" + set_name + "-grammar-heatmap"
        annotation_file_name: str = self.output_dir + "/" + set_name + "-annotation.txt"

        if os.path.isfile(annotation_file_name) and \
                not os.path.isfile(file_name + ".pdf"):
            df = pd.read_csv(annotation_file_name, sep="\t",
                             dtype={"annotation": "string", "y": "string"})
            df = df.fillna("")
            counts: Dict[str, List[int]] = dict()
            annotations: List[str] = df["annotation"].tolist()
            if self.definition.task == c.TaskType.MULTI_CLASS_CLASSIFICATION:
                y: List[str] = df["y"].tolist()
            elif self.definition.task == c.TaskType.MULTI_LABEL_CLASSIFICATION:
                y: List[str] = df["y"].replace(np.nan, "", regex=True).tolist()

            for i, annotation in enumerate(annotations):
                labels: List[str] = None
                if self.definition.task == c.TaskType.MULTI_CLASS_CLASSIFICATION:
                    labels = [y[i]]
                elif self.definition.task == c.TaskType.MULTI_LABEL_CLASSIFICATION:
                    if y[i] == "":
                        labels = ["none (background)"]
                    else:
                        labels = y[i].split("|")

                for label in labels:
                    if label not in counts:
                        counts[label] = [0] * \
                            self.definition.background.max_length

                    for j, letter in enumerate(annotation):
                        if letter == c.PositionType.GRAMMAR:
                            counts[label][j] += 1

            label_column: List[str] = list()
            position_column: List[int] = list()
            grammar_probability_column: List[float] = list()

            for label, grammar_counts in counts.items():
                for i, grammar_count in enumerate(grammar_counts):
                    label_column.append(label)
                    position_column.append(i + 1)
                    grammar_probability_column.append(
                        grammar_count / len(annotations))

            plot_df = pd.DataFrame(
                {"label": label_column,
                 "position": position_column,
                 "grammar_probability": grammar_probability_column})
            plot_df.to_csv(file_name + ".txt", sep="\t", index=False)

            plot_script: str = pkg_resources.resource_filename(
                "seqgra", "simulator/plotgrammarpositionheatmap.R")

            cmd = ["Rscript", "--no-save", "--no-restore", "--quiet",
                   plot_script, file_name + ".txt", file_name + ".pdf"]

            try:
                subprocess.call(cmd, universal_newlines=True)
            except subprocess.CalledProcessError as exception:
                self.logger.warning("failed to create grammar position "
                                    "heatmaps: %s", exception.output)
            except FileNotFoundError as exception:
                self.logger.warning("Rscript not on PATH, skipping "
                                    "grammar position heatmaps")

    def __add_shuffled_examples(self, set_name: str,
                                preserve_frequencies_for_kmer: int,
                                labels_value: str) -> None:
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
                        "".join([c.PositionType.BACKGROUND] * len(columns[0])) +
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
                                        preserve_frequencies_for_kmer),
                       "utf-8")
        else:
            example_list = list(example)
            random.shuffle(example_list)
            return "".join(example_list)

    def write_session_info(self) -> None:
        with open(self.output_dir + "session-info.txt", "w") as session_file:
            session_file.write(
                "seqgra package version: " +
                pkg_resources.require("seqgra")[0].version + "\n")
            session_file.write("NumPy version: " + np.version.version + "\n")
            session_file.write("Python version: " + sys.version + "\n")

    def __process_set(self, example_set: DataGenerationSet) -> None:
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
                example: AnnotatedExample = ExampleGenerator.generate_example(
                    conditions, example_set.name, self.definition.background)
                data_file.write(example.x + "\t" + condition_id + "\n")
                annotation_file.write(
                    example.annotation + "\t" + condition_id + "\n")

    def __serialize_example(self, example: DataGenerationExample) -> str:
        return "|".join([condition.condition_id
                         for condition in example.conditions])

    def __deserialize_example(self, example_str_rep: str) -> List[Condition]:
        condition_ids: List[str] = example_str_rep.split("|")
        return [Condition.get_by_id(self.definition.conditions, condition_id)
                for condition_id in condition_ids]

    def __set_seed(self) -> None:
        random.seed(self.definition.seed)
        np.random.seed(self.definition.seed)
        ushuffle.set_seed(self.definition.seed)

    def check_grammar(self) -> bool:
        valid: bool = True

        c1: bool = self.check_unused_conditions()
        c2: bool = self.check_unused_sequence_elements()
        c3: bool = self.check_missing_alphabet_distributions()
        c4: bool = self.check_invalid_positions()
        c5: bool = self.check_invalid_distances()
        c6: bool = self.check_invalid_sequence_elements()
        c7: bool = self.check_spacing_contraint_se_refs()
        c8: bool = self.check_overlapping_sequence_elements()

        valid = c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8
        if valid:
            self.logger.info("semantic analysis of grammar completed: "
                             "no issues detected")
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
                self.logger.warning("condition %s [cid]: unused condition",
                                    condition.condition_id)

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
                self.logger.warning("sequence element %s [sid]: unused "
                                    "sequence element", sequence_element.sid)

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
                            self.logger.warning("more than one global "
                                                "alphabet definition found")
                        else:
                            set_condition_combinations[set_name][condition_id] = "global"
            elif alphabet.condition_independent:
                tmp_dict = set_condition_combinations[alphabet.set_name]
                for condition_id in tmp_dict.keys():
                    if tmp_dict[condition_id] == "condition-independent":
                        valid = False
                        self.logger.warning("more than one "
                                            "condition-independent alphabet "
                                            "definition found for set %s",
                                            alphabet.set_name)
                    else:
                        tmp_dict[condition_id] = "condition-independent"
            elif alphabet.set_independent:
                for set_name in set_condition_combinations.keys():
                    if set_condition_combinations[set_name][alphabet.condition.condition_id] == "set-independent":
                        valid = False
                        self.logger.warning("more than one set-independent "
                                            "alphabet definition found for "
                                            "condition %s [cid]",
                                            alphabet.condition.condition_id)
                    else:
                        set_condition_combinations[set_name][alphabet.condition.condition_id] = "set-independent"
            else:
                if set_condition_combinations[alphabet.set_name][alphabet.condition.condition_id] == "specified":
                    valid = False
                    self.logger.warning("duplicate alphabet definition found "
                                        "for set name %s and condition "
                                        "%s [cid]",
                                        alphabet.set_name, alphabet.condition_id)
                else:
                    set_condition_combinations[alphabet.set_name][alphabet.condition.condition_id] = "specified"

        for set_name, tmp_dict in set_condition_combinations.items():
            for condition_id, value in tmp_dict.items():
                if value == "unspecified":
                    valid = False
                    self.logger.warning("no alphabet definition found for set "
                                        "name %s and condition %s [cid]",
                                        set_name, condition_id)

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
                        self.logger.warning("condition %s [cid], rule %s: "
                                            "position exceeds minimum "
                                            "sequence length",
                                            condition.condition_id, i + 1)
                    elif int(rule.probability) + self.__get_longest_sequence_element_length(rule) > self.definition.background.min_length:
                        valid = False
                        self.logger.warning("condition %s [cid], rule %s: "
                                            "position plus sequence element "
                                            "length exceeds minimum sequence "
                                            "length",
                                            condition.condition_id, i + 1)
        return valid

    def check_overlapping_sequence_elements(self) -> bool:
        valid: bool = True
        for condition in self.definition.conditions:
            for i in range(len(condition.grammar)):
                rule = condition.grammar[i]
                if rule.position != "random" and \
                    len(rule.sequence_elements) > 1 and \
                        not rule.spacing_constraints:
                    valid = False
                    self.logger.warning("condition %s [cid], rule %s: "
                                        "overlapping sequence elements "
                                        "(possible solutions: (1) position "
                                        "randomly, (2) add spacing "
                                        "constraint, (3) split in two rules "
                                        "with different positions)",
                                        condition.condition_id, i + 1)

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
                            self.logger.warning("condition %s [cid], rule %s, "
                                                "spacing constraint %s: "
                                                "minimum  distance exceeds "
                                                "minimum sequence length",
                                                condition.condition_id,
                                                i + 1, j + 1)
                        elif spacing_constraint.min_distance + spacing_constraint.sequence_element1.get_max_length() + spacing_constraint.sequence_element2.get_max_length() > self.definition.background.min_length:
                            valid = False
                            self.logger.warning("condition %s [cid], rule %s, "
                                                "spacing constraint %s: "
                                                "minimum distance plus "
                                                "sequence element lengths "
                                                "exceeds minimum "
                                                "sequence length",
                                                condition.condition_id,
                                                i + 1, j + 1)

        return valid

    def check_invalid_sequence_elements(self) -> bool:
        valid: bool = True

        for sequence_element in self.definition.sequence_elements:
            if sequence_element.get_max_length() > self.definition.background.min_length:
                valid = False
                self.logger.warning("sequence element %s: maximum sequence "
                                    "element length exceeds minimum sequence "
                                    "length", sequence_element.sid)

        return valid

    def check_spacing_contraint_se_refs(self) -> bool:
        valid: bool = True
        valid_sequence_element_ids: Set[str] = set()
        for condition in self.definition.conditions:
            for i in range(len(condition.grammar)):
                rule: Rule = condition.grammar[i]
                if rule.spacing_constraints is not None and \
                        len(rule.spacing_constraints) > 0:
                    valid_sequence_element_ids.clear()
                    for sequence_element in rule.sequence_elements:
                        valid_sequence_element_ids.add(sequence_element.sid)

                    for j in range(len(rule.spacing_constraints)):
                        spacing_constraint: SpacingConstraint = rule.spacing_constraints[j]
                        if spacing_constraint.sequence_element1.sid not in valid_sequence_element_ids:
                            valid = False
                            self.logger.error("condition %s [cid], rule %s, "
                                              "spacing constraint %s: "
                                              "sequence  element %s [sid] "
                                              "not among "
                                              "sequence elements of rule",
                                              condition.condition_id,
                                              i + 1, j + 1,
                                              spacing_constraint.sequence_element1.sid)
                        if spacing_constraint.sequence_element2.sid not in valid_sequence_element_ids:
                            valid = False
                            self.logger.error("condition %s [cid], rule %s, "
                                              "spacing constraint %s: "
                                              "sequence element %s [sid] "
                                              "not among "
                                              "sequence elements of rule",
                                              condition.condition_id,
                                              i + 1, j + 1,
                                              spacing_constraint.sequence_element2.sid)

        return valid

    def __get_longest_sequence_element_length(self, rule: Rule):
        longest_length: int = 0

        for sequence_element in rule.sequence_elements:
            if sequence_element.get_max_length() > longest_length:
                longest_length = sequence_element.get_max_length()
        return longest_length
