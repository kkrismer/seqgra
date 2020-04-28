"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd

from seqgra.learner import Learner
from seqgra.evaluator import Evaluator
from seqgra.evaluator.sis import sis_collection
from seqgra.evaluator.sis import make_empty_boolean_mask_broadcast_over_axis
from seqgra.evaluator.sis import produce_masked_inputs


class PositionClass:
    GRAMMAR: str = "G"
    BACKGROUND: str = "_"
    CONFOUNDER: str = "C"
    DNA_MASKED: str = "N"
    AA_MASKED: str = "X"


class SISEvaluator(Evaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("sis", learner, output_dir)

    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        # TODO get selected labels, not all
        labels: List[str] = self.learner.definition.labels

        # TODO use class attribute threshold
        threshold = 0.5

        x_column: List[str] = list()
        y_column: List[str] = list()
        annotations_column: List[str] = list()
        sis_collapsed_column: List[str] = list()
        precision_column: List[str] = list()
        recall_column: List[str] = list()
        sis_separated_column: List[str] = list()
        for i, selected_label in enumerate(labels):
            # select x, y, annotations of examples with label
            subset_idx = [i
                          for i, label in enumerate(y)
                          if label == selected_label]
            selected_x, selected_y, selected_annotations = \
                self._subset(subset_idx, x, y, annotations)

            sis_results: List[List[str]] = self.find_sis(
                selected_x, threshold, i)
            sis_collapsed: List[str] = [self.__collapse_sis(sis_col)
                                        for sis_col in sis_results]
            precision: List[float] = [self.calculate_precision(sis, anno)
                                      for sis, anno in zip(sis_collapsed,
                                                           selected_annotations)]
            recall: List[float] = [self.calculate_recall(sis, anno)
                                   for sis, anno in zip(sis_collapsed,
                                                        selected_annotations)]
            sis_separated: List[str] = [";".join(sis_col)
                                        for sis_col in sis_results]

            x_column += selected_x
            y_column += selected_y
            annotations_column += selected_annotations
            sis_collapsed_column += sis_collapsed
            precision_column += precision
            recall_column += recall
            sis_separated_column += sis_separated

        return pd.DataFrame({"x": x_column,
                             "y": y_column,
                             "annotation": annotations_column,
                             "sis_collapsed": sis_collapsed_column,
                             "precision": precision_column,
                             "recall": recall_column,
                             "sis_separated": sis_separated_column})

    def _save_results(self, results, set_name: str = "test") -> None:
        if results is None:
            results = pd.DataFrame([], columns=["x", "y", "annotation",
                                                "sis_collapsed", "precision",
                                                "recall", "sis_separated"])

        results.to_csv(self.output_dir + set_name + ".txt", sep="\t",
                       index=False)

    def find_sis(self, x: List[str], threshold: float,
                 label_index: int) -> List[List[str]]:
        encoded_x = self.learner.encode_x(x)

        def sis_predict(x):
            return np.array(self.learner.predict(
                x, encode=False))[:, label_index]

        input_shape = encoded_x[0].shape
        fully_masked_input = np.ones(input_shape) * 0.25
        initial_mask = make_empty_boolean_mask_broadcast_over_axis(
            input_shape, 1)

        return [self.__produce_masked_inputs(
            encoded_x[i], sis_predict, threshold,
            fully_masked_input, initial_mask)
            for i in range(len(encoded_x))]

    def __calculate_precision(self, sis: str, annotation: str) -> float:
        if self.learner.definition.sequence_space == "DNA":
            masked_letter: str = PositionClass.DNA_MASKED
        else:
            masked_letter: str = PositionClass.AA_MASKED

        if sis == "":
            return 1.0
        else:
            num_selected: int = 0
            num_selected_relevant: int = 0
            for i, c in enumerate(sis):
                if c != masked_letter:
                    num_selected += 1
                    if annotation[i] == PositionClass.GRAMMAR:
                        num_selected_relevant += 1

            if num_selected == 0:
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
        if self.learner.definition.sequence_space == "DNA":
            masked_letter: str = PositionClass.DNA_MASKED
        else:
            masked_letter: str = PositionClass.AA_MASKED

        num_relevent: int = 0

        if sis == "":
            for i, c in enumerate(annotation):
                if c == PositionClass.GRAMMAR:
                    num_relevent += 1
            if num_relevent == 0:
                return 1.0
            else:
                return 0.0
        else:
            num_relevant_selected: int = 0
            for i, c in enumerate(annotation):
                if c == PositionClass.GRAMMAR:
                    num_relevent += 1
                    if sis[i] != masked_letter:
                        num_relevant_selected += 1

            if num_relevent == 0:
                return 1.0

            return num_relevant_selected / num_relevent

    def calculate_recall(self, sis: List[str],
                         annotations: List[str]) -> List[float]:
        if sis is None:
            return list()
        else:
            return [self.__calculate_recall(s, anno)
                    for s, anno in zip(sis, annotations)]

    def __collapse_sis(self, sis: List[str]) -> str:
        if self.learner.definition.sequence_space == "DNA":
            masked_letter: str = PositionClass.DNA_MASKED
        else:
            masked_letter: str = PositionClass.AA_MASKED

        if len(sis) == 0:
            return ""
        elif len(sis) == 1:
            return sis[0]
        else:
            collapsed_sis: str = sis[0]
            sis.pop(0)
            for i, c in enumerate(collapsed_sis):
                if c == masked_letter:
                    for s in sis:
                        if s[i] != masked_letter:
                            collapsed_sis = collapsed_sis[:i] + \
                                s[i] + collapsed_sis[(i + 1):]
            return collapsed_sis

    def __produce_masked_inputs(self, x, sis_predict, threshold,
                                fully_masked_input,
                                initial_mask) -> List[str]:
        collection = sis_collection(sis_predict, threshold, x,
                                    fully_masked_input,
                                    initial_mask=initial_mask)

        if len(collection) > 0:
            sis_masked_inputs = produce_masked_inputs(
                x, fully_masked_input,
                [sr.mask for sr in collection])
            return self.learner.decode_x(sis_masked_inputs).tolist()
        else:
            return list()
