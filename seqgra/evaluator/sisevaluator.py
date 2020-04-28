"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import os
import pandas as pd
import pkg_resources
import subprocess

import seqgra.constants as c
from seqgra.evaluator import Evaluator
from seqgra.evaluator.sis import make_empty_boolean_mask_broadcast_over_axis
from seqgra.evaluator.sis import produce_masked_inputs
from seqgra.evaluator.sis import sis_collection
from seqgra.learner import Learner


class SISEvaluator(Evaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(
            "sis", learner, output_dir,
            supported_tasks=[c.TaskType.MULTI_CLASS_CLASSIFICATION])

    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        x_column: List[str] = list()
        y_column: List[str] = list()
        annotations_column: List[str] = list()
        sis_collapsed_column: List[str] = list()
        precision_column: List[str] = list()
        recall_column: List[str] = list()
        sensitivity_column: List[str] = list()
        specificity_column: List[str] = list()
        sis_separated_column: List[str] = list()
        for i, selected_label in enumerate(set(y)):
            # select x, y, annotations of examples with label
            subset_idx = [i
                          for i, label in enumerate(y)
                          if label == selected_label]
            selected_x, selected_y, selected_annotations = \
                self._subset(subset_idx, x, y, annotations)

            sis_results: List[List[str]] = self.find_sis(
                selected_x, i)
            sis_collapsed: List[str] = [self.__collapse_sis(sis_col)
                                        for sis_col in sis_results]
            precision: List[float] = self.calculate_precision(
                sis_collapsed, selected_annotations)
            recall: List[float] = self.calculate_recall(
                sis_collapsed, selected_annotations)
            sensitivity: List[float] = self.calculate_sensitivity(
                sis_collapsed, selected_annotations)
            specificity: List[float] = self.calculate_specificity(
                sis_collapsed, selected_annotations)
            sis_separated: List[str] = [";".join(sis_col)
                                        for sis_col in sis_results]

            x_column += selected_x
            y_column += selected_y
            annotations_column += selected_annotations
            sis_collapsed_column += sis_collapsed
            precision_column += precision
            recall_column += recall
            sensitivity_column += sensitivity
            specificity_column += specificity
            sis_separated_column += sis_separated

        return pd.DataFrame({"x": x_column,
                             "y": y_column,
                             "annotation": annotations_column,
                             "sis_collapsed": sis_collapsed_column,
                             "precision": precision_column,
                             "recall": recall_column,
                             "sensitivity": sensitivity_column,
                             "specificity": specificity_column,
                             "sis_separated": sis_separated_column})

    def _save_results(self, results, set_name: str = "test") -> None:
        if results is None:
            results = pd.DataFrame([], columns=["x", "y", "annotation",
                                                "sis_collapsed", "precision",
                                                "recall", "sis_separated"])

        results.to_csv(self.output_dir + set_name + ".txt", sep="\t",
                       index=False)
        if self.create_plots and len(results.index) > 0:
            self.visualize_agreement(results, set_name)

    def __get_agreement_group(self, annotation_position: str,
                              sis_position: str) -> str:
        masked_letter: str = self.__get_masked_letter()

        if annotation_position == c.PositionType.GRAMMAR:
            if sis_position == masked_letter:
                return "FN (grammar position, not part of SIS)"
            else:
                return "TP (grammar position, part of SIS)"
        else:
            if sis_position == masked_letter:
                return "TN (background position, not part of SIS)"
            else:
                return "FP (background position, part of SIS)"

    def __prepare_r_data_frame(self, results, file_name):
        example_column: List[int] = list()
        position_column: List[int] = list()
        group_column: List[int] = list()
        label_column: List[int] = list()
        precision_column: List[int] = list()
        recall_column: List[int] = list()
        sensitivity_column: List[int] = list()
        specificity_column: List[int] = list()
        n_column: List[float] = list()

        for example_id, row in enumerate(results.itertuples(), 1):
            example_column += [example_id] * len(row.annotation)
            position_column += list(range(1, len(row.annotation) + 1))
            if row.sis_collapsed:
                group_column += [self.__get_agreement_group(char, row.sis_collapsed[i])
                                 for i, char in enumerate(row.annotation)]
            else:
                group_column += [self.__get_agreement_group(char, self.__get_masked_letter())
                                 for i, char in enumerate(row.annotation)]
            label_column += [row.y] * len(row.annotation)
            precision_column += [row.precision] * len(row.annotation)
            recall_column += [row.recall] * len(row.annotation)
            sensitivity_column += [row.sensitivity] * len(row.annotation)
            specificity_column += [row.specificity] * len(row.annotation)
            n_column += [1.0 / len(row.annotation)] * len(row.annotation)

        df = pd.DataFrame({"example": example_column,
                           "position": position_column,
                           "group": group_column,
                           "label": label_column,
                           "precision": precision_column,
                           "recall": recall_column,
                           "sensitivity": sensitivity_column,
                           "specificity": specificity_column,
                           "n": n_column})
        df["precision"] = df.groupby("label")["precision"].transform("mean")
        df["recall"] = df.groupby("label")["recall"].transform("mean")
        df["sensitivity"] = df.groupby(
            "label")["sensitivity"].transform("mean")
        df["specificity"] = df.groupby(
            "label")["specificity"].transform("mean")
        df["n"] = round(df.groupby("label")["n"].transform("sum"))
        df.to_csv(file_name, sep="\t", index=False)

    def visualize_agreement(self, results, set_name: str = "test") -> None:
        plot_script: str = pkg_resources.resource_filename(
            "seqgra", "evaluator/sis/plotsis.R")
        temp_file_name: str = self.output_dir + set_name + "-temp.txt"
        pdf_file_name: str = self.output_dir + set_name + "-agreement.pdf"
        self.__prepare_r_data_frame(results, temp_file_name)

        cmd = ["Rscript", plot_script, temp_file_name, pdf_file_name]
        subprocess.check_output(cmd, universal_newlines=True)
        os.remove(temp_file_name)

    def find_sis(self, x: List[str], label_index: int) -> List[List[str]]:
        encoded_x = self.learner.encode_x(x)

        def sis_predict(x):
            return np.array(self.learner.predict(
                x, encode=False))[:, label_index]

        input_shape = encoded_x[0].shape
        fully_masked_input = np.ones(input_shape) * 0.25
        initial_mask = make_empty_boolean_mask_broadcast_over_axis(
            input_shape, 1)

        return [self.__produce_masked_inputs(
            encoded_x[i], sis_predict,
            fully_masked_input, initial_mask)
            for i in range(len(encoded_x))]

    def __get_masked_letter(self) -> str:
        if self.learner.definition.sequence_space == c.SequenceSpaceType.DNA:
            return c.PositionType.DNA_MASKED
        else:
            return c.PositionType.AA_MASKED

    def __calculate_precision(self, sis: str, annotation: str) -> float:
        masked_letter: str = self.__get_masked_letter()

        if not sis:
            return 1.0
        else:
            num_selected: int = 0
            num_selected_relevant: int = 0
            for i, char in enumerate(sis):
                if char != masked_letter:
                    num_selected += 1
                    if annotation[i] == c.PositionType.GRAMMAR:
                        num_selected_relevant += 1

            if not num_selected:
                return 1.0

            return num_selected_relevant / num_selected

    def calculate_precision(self, sis: List[str],
                            annotations: List[str]) -> List[float]:
        if sis is None:
            return list()
        else:
            return [self.__calculate_precision(s, anno)
                    for s, anno in zip(sis, annotations)]

    def __calculate_recall(self, sis: str, annotation: str) -> float:
        masked_letter: str = self.__get_masked_letter()
        num_relevent: int = 0

        if not sis:
            for i, char in enumerate(annotation):
                if char == c.PositionType.GRAMMAR:
                    num_relevent += 1
            if not num_relevent:
                return 1.0
            else:
                return 0.0
        else:
            num_relevant_selected: int = 0
            for i, char in enumerate(annotation):
                if char == c.PositionType.GRAMMAR:
                    num_relevent += 1
                    if sis[i] != masked_letter:
                        num_relevant_selected += 1

            if not num_relevent:
                return 1.0

            return num_relevant_selected / num_relevent

    def calculate_recall(self, sis: List[str],
                         annotations: List[str]) -> List[float]:
        if sis is None:
            return list()
        else:
            return [self.__calculate_recall(s, anno)
                    for s, anno in zip(sis, annotations)]

    def __calculate_sensitivity(self, sis: str, annotation: str) -> float:
        masked_letter: str = self.__get_masked_letter()
        num_true_positive: int = 0
        num_false_negative: int = 0

        if not sis:
            for i, char in enumerate(annotation):
                if char == c.PositionType.GRAMMAR:
                    num_false_negative += 1
            if not num_false_negative:
                return 1.0
            else:
                return 0.0
        else:
            for i, char in enumerate(annotation):
                if char == c.PositionType.GRAMMAR:
                    if sis[i] == masked_letter:
                        num_false_negative += 1
                    else:
                        num_true_positive += 1

        if not num_true_positive and not num_false_negative:
            return 0.0
        else:
            return num_true_positive / (num_true_positive + num_false_negative)

    def calculate_sensitivity(self, sis: List[str],
                              annotations: List[str]) -> List[float]:
        if sis is None:
            return list()
        else:
            return [self.__calculate_sensitivity(s, anno)
                    for s, anno in zip(sis, annotations)]

    def __calculate_specificity(self, sis: str, annotation: str) -> float:
        masked_letter: str = self.__get_masked_letter()
        num_true_negative: int = 0
        num_false_positive: int = 0

        if not sis:
            for i, char in enumerate(annotation):
                if char == c.PositionType.BACKGROUND or \
                        char == c.PositionType.CONFOUNDER:
                    num_true_negative += 1
            if num_true_negative > 0:
                return 1.0
            else:
                return 0.0
        else:
            for i, char in enumerate(annotation):
                if char == c.PositionType.BACKGROUND or \
                        char == c.PositionType.CONFOUNDER:
                    if sis[i] == masked_letter:
                        num_true_negative += 1
                    else:
                        num_false_positive += 1

        if not num_true_negative and not num_false_positive:
            return 0.0
        else:
            return num_true_negative / (num_true_negative + num_false_positive)

    def calculate_specificity(self, sis: List[str],
                              annotations: List[str]) -> List[float]:
        if sis is None:
            return list()
        else:
            return [self.__calculate_specificity(s, anno)
                    for s, anno in zip(sis, annotations)]

    def __collapse_sis(self, sis: List[str]) -> str:
        masked_letter: str = self.__get_masked_letter()

        if not sis:
            return ""
        elif len(sis) == 1:
            return sis[0]
        else:
            collapsed_sis: str = sis[0]
            sis.pop(0)
            for i, char in enumerate(collapsed_sis):
                if char == masked_letter:
                    for s in sis:
                        if s[i] != masked_letter:
                            collapsed_sis = collapsed_sis[:i] + \
                                s[i] + collapsed_sis[(i + 1):]
            return collapsed_sis

    def __produce_masked_inputs(self, x, sis_predict, fully_masked_input,
                                initial_mask) -> List[str]:
        collection = sis_collection(sis_predict, self.threshold, x,
                                    fully_masked_input,
                                    initial_mask=initial_mask)

        if len(collection) > 0:
            sis_masked_inputs = produce_masked_inputs(
                x, fully_masked_input,
                [sr.mask for sr in collection])
            return self.learner.decode_x(sis_masked_inputs).tolist()
        else:
            return list()
