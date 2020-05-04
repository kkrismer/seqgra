"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod
import logging
import os
import random
import pkg_resources
import subprocess
from typing import Any, List, Optional, Set

import numpy as np
import pandas as pd

import seqgra.constants as c
from seqgra import AnnotatedExampleSet
from seqgra import MiscHelper
from seqgra.learner import Learner


class Evaluator(ABC):
    @abstractmethod
    def __init__(self, evaluator_id: str, evaluator_name: str,
                 learner: Learner,
                 output_dir: str,
                 supported_tasks: Optional[Set[str]] = None,
                 supported_sequence_spaces: Optional[Set[str]] = None,
                 supported_libraries: Optional[Set[str]] = None) -> None:
        self.evaluator_id: str = evaluator_id
        self.evaluator_name: str = evaluator_name
        self.learner: Learner = learner
        self.output_dir = MiscHelper.prepare_path(output_dir + "/" +
                                                  self.evaluator_id,
                                                  allow_exists=False)
        if supported_tasks is None:
            self.supported_tasks: Set[str] = c.TaskType.ALL_TASKS
        else:
            self.supported_tasks: Set[str] = supported_tasks
        if supported_sequence_spaces is None:
            self.supported_sequence_spaces: Set[str] = c.SequenceSpaceType.ALL_SEQUENCE_SPACES
        else:
            self.supported_sequence_spaces: Set[str] = supported_sequence_spaces
        if supported_libraries is None:
            self.supported_libraries: Set[str] = c.LibraryType.ALL_LIBRARIES
        else:
            self.supported_libraries: Set[str] = supported_libraries
        self.__detect_incompatibilities()

    def __detect_incompatibilities(self, throw_exception: bool = True):
        message: str = ""
        if not self.learner.definition.task in self.supported_tasks:
            message: str = "learner task incompatible with evaluator"
        if not self.learner.definition.sequence_space in self.supported_sequence_spaces:
            message: str = "learner sequence space incompatible with evaluator"
        if not self.learner.definition.library in self.supported_libraries:
            message: str = "learner library incompatible with evaluator"

        if message:
            if throw_exception:
                raise Exception(message)
            logging.warning(message)

    def evaluate_model(self, set_name: str = "test",
                       subset_idx: Optional[List[int]] = None,
                       subset_n: Optional[int] = None,
                       subset_labels: Optional[List[str]] = None,
                       subset_n_per_label: bool = True,
                       subset_shuffle: bool = True,
                       subset_threshold: Optional[float] = None,
                       suppress_plots: bool = False) -> Any:
        if subset_idx:
            x, y, annotations = self._load_data(set_name, subset_idx)
        elif subset_n or subset_labels or \
                subset_n_per_label or subset_threshold:
            x, y, annotations = self.select_n_examples(set_name, subset_n,
                                                       subset_labels,
                                                       subset_n_per_label,
                                                       subset_shuffle,
                                                       subset_threshold)
        else:
            x, y, annotations = self._load_data(set_name)

        results = self._evaluate_model(x, y, annotations)
        self._save_results(results, set_name, suppress_plots)

        return results

    def _load_data(self, set_name: str = "test",
                   subset_idx: Optional[List[int]] = None) -> AnnotatedExampleSet:
        # load data
        examples_file: str = self.learner.get_examples_file(set_name)
        annotations_file: str = self.learner.get_annotations_file(set_name)
        x, y = self.learner.parse_examples_data(examples_file)
        annotations, _ = self.learner.parse_annotations_data(annotations_file)

        if subset_idx is None:
            return AnnotatedExampleSet(x, y, annotations)
        else:
            return self._subset(subset_idx, x, y, annotations)

    @abstractmethod
    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        pass

    @abstractmethod
    def _save_results(self, results, set_name: str = "test",
                      suppress_plots: bool = False) -> None:
        pass

    @staticmethod
    def _subset(idx: List[int], x: List[str], y: List[str],
                annotations: List[str]) -> AnnotatedExampleSet:
        if len(x) != len(y) or len(x) != len(annotations):
            raise Exception("x, y, and annotations have to be the same length")

        if max(idx) >= len(x):
            raise Exception("max(idx) >= len(x)")

        x = [x[i] for i in idx]
        y = [y[i] for i in idx]
        annotations = [annotations[i] for i in idx]

        return AnnotatedExampleSet(x, y, annotations)

    def _subset_by_label(self, x: List[str], y: List[str],
                         annotations: List[str],
                         valid_labels: Set[str]) -> AnnotatedExampleSet:
        if self.learner.definition.task == c.TaskType.MULTI_CLASS_CLASSIFICATION:
            subset_idx = [i
                          for i, label in enumerate(y)
                          if label in valid_labels]
        elif self.learner.definition.task == c.TaskType.MULTI_LABEL_CLASSIFICATION:
            subset_idx: List[int] = list()
            for i, label in enumerate(y):
                is_valid: bool = False
                for example_label in label.split("|"):
                    if example_label in valid_labels:
                        is_valid = True
                if is_valid:
                    subset_idx += [i]

        return self._subset(subset_idx, x, y, annotations)

    def _subset_by_compatibility(
            self, x: List[str], y: List[str],
            annotations: List[str]) -> AnnotatedExampleSet:
        return AnnotatedExampleSet(x, y, annotations)

    def select_examples(self, set_name: str = "test",
                        labels: Optional[Set[str]] = None,
                        threshold: Optional[float] = None) -> AnnotatedExampleSet:
        """Returns all correctly classified examples that exceed the threshold.

        for the specified labels
        and set that exceed the threshold.

        Parameters:
            TODO

        Returns:
            TODO
        """
        examples_file: str = self.learner.get_examples_file(set_name)
        annotations_file: str = self.learner.get_annotations_file(set_name)
        x, y = self.learner.parse_examples_data(examples_file)
        annotations, _ = self.learner.parse_annotations_data(annotations_file)

        if labels is not None:
            # discard examples with y not in labels
            x, y, annotations = self._subset_by_label(x, y, annotations,
                                                      labels)

        if threshold:
            # predict with learner
            encoded_y = self.learner.encode_y(y)
            y_hat = self.learner.predict(x)

            # discard misclassified / mislabeled examples and
            # examples below threshold
            if self.learner.definition.task == c.TaskType.MULTI_CLASS_CLASSIFICATION:
                subset_idx = [i
                              for i in range(len(x))
                              if np.argmax(y_hat[i]) == np.argmax(encoded_y[i]) and
                              np.max(y_hat[i]) > threshold]
            elif self.learner.definition.task == c.TaskType.MULTI_LABEL_CLASSIFICATION:
                subset_idx: List[int] = list()
                # example is mislabeled if prediction for one label is wrong
                # correctly labeled example:
                # if label == True -> prediction > threshold
                # if label == False -> prediction < (1 - threshold)
                for example_index in range(len(x)):
                    is_valid: bool = True
                    for label_index in range(len(self.learner.definition.labels)):
                        if (encoded_y[example_index, label_index] and
                            y_hat[example_index, label_index] < threshold) or \
                                (not encoded_y[example_index, label_index] and
                                    y_hat[example_index, label_index] > (1 - threshold)):
                            is_valid = False

                    if is_valid:
                        subset_idx += [example_index]

            x, y, annotations = self._subset(subset_idx, x, y, annotations)

        x, y, annotations = self._subset_by_compatibility(x, y, annotations)

        return AnnotatedExampleSet(x, y, annotations)

    def select_n_examples(self, set_name: str = "test",
                          n: Optional[int] = None,
                          labels: Optional[Set[str]] = None,
                          n_per_label: bool = True,
                          shuffle: bool = True,
                          threshold: Optional[float] = None) -> AnnotatedExampleSet:
        if n_per_label:
            if labels is None:
                labels = self.learner.definition.labels
            x: List[str] = list()
            y: List[str] = list()
            annotations: List[str] = list()
            for label in labels:
                examples = self.select_n_examples(set_name, n, label, False,
                                                  shuffle, threshold)
                x += examples.x
                y += examples.y
                annotations += examples.annotations
            return AnnotatedExampleSet(x, y, annotations)
        else:
            x, y, annotations = self.select_examples(set_name, labels,
                                                     threshold)
            if n is None:
                n = len(x)

            if not x:
                if labels is None and threshold is None:
                    logging.warning("no example in set '%s'", set_name)
                elif labels is None and threshold is not None:
                    logging.warning("no correctly labeled example with "
                                    "prediction threshold > %s in set '%s'",
                                    threshold, set_name)
                elif labels is not None and threshold is None:
                    logging.warning("no example in set '%s' "
                                    "that has one of the following labels: %s",
                                    set_name, labels)
                elif labels is not None and threshold is not None:
                    logging.warning("no correctly labeled example with "
                                    "prediction threshold > %s in set '%s' "
                                    "that has one of the following labels: %s",
                                    threshold, set_name, labels)
                return AnnotatedExampleSet(None, None, None)
            elif n > len(x):
                n = len(x)

            if shuffle:
                subset_idx: List[int] = list(range(len(x)))
                random.shuffle(subset_idx)
                subset_idx = subset_idx[:n]
            else:
                subset_idx: List[int] = list(range(n))

            return self._subset(subset_idx, x, y, annotations)


class FeatureImportanceEvaluator(Evaluator):
    def __init__(self, evaluator_id: str, evaluator_name: str,
                 learner: Learner,
                 output_dir: str,
                 supported_tasks: Optional[Set[str]] = None,
                 supported_sequence_spaces: Optional[Set[str]] = None,
                 supported_libraries: Optional[Set[str]] = None) -> None:
        super().__init__(evaluator_id, evaluator_name, learner, output_dir,
                         supported_tasks,
                         supported_sequence_spaces,
                         supported_libraries)

    def evaluate_model(self, set_name: str = "test",
                       subset_idx: Optional[List[int]] = None,
                       subset_n: Optional[int] = None,
                       subset_labels: Optional[List[str]] = None,
                       subset_n_per_label: bool = True,
                       subset_shuffle: bool = True,
                       subset_threshold: Optional[float] = None,
                       suppress_plots: bool = False) -> Any:
        results = super().evaluate_model(set_name, subset_idx, subset_n,
                                         subset_labels, subset_n_per_label,
                                         subset_shuffle, subset_threshold,
                                         suppress_plots)
        if not suppress_plots:
            self._visualize_grammar_agreement(results, set_name)
        return results

    def _subset_by_compatibility(
            self, x: List[str], y: List[str],
            annotations: List[str]) -> AnnotatedExampleSet:
        # remove examples without positive labels
        if self.learner.definition.task == c.TaskType.MULTI_LABEL_CLASSIFICATION:
            subset_idx = [i
                          for i, label in enumerate(y)
                          if label]
            return self._subset(subset_idx, x, y, annotations)
        else:
            return AnnotatedExampleSet(x, y, annotations)

    @abstractmethod
    def _convert_to_data_frame(self, results) -> pd.DataFrame:
        """Takes evaluator-specific results and turns them into a pandas
        data frame.

        The data frame must have at least the following columns:
            - example_column (int): example index
            - position (int): position within example (one-based)
            - group (str): group label, one of the following:
                - "TP": grammar position, important for model prediction
                - "FN": grammar position, not important for model prediction,
                - "FP": background position, important for model prediction,
                - "TN": background position, not important for model prediction
            - label (str): label of example, e.g., "cell type 1"
        """

    @staticmethod
    def _calculate_precision(positions: List[str]) -> float:
        num_true_positive: int = positions.count("TP")
        num_false_positive: int = positions.count("FP")
        num_false_negative: int = positions.count("FN")

        if not num_true_positive and not num_false_positive and \
                not num_false_negative:
            return 1.0
        elif not num_true_positive and not num_false_positive:
            return 0.0
        else:
            return num_true_positive / (num_true_positive + num_false_positive)

    @staticmethod
    def _calculate_recall(positions: List[str]) -> float:
        num_true_positive: int = positions.count("TP")
        num_false_positive: int = positions.count("FP")
        num_false_negative: int = positions.count("FN")

        if not num_true_positive and not num_false_positive and \
                not num_false_negative:
            return 1.0
        elif not num_true_positive and not num_false_negative:
            return 0.0
        else:
            return num_true_positive / (num_true_positive + num_false_negative)

    @staticmethod
    def _calculate_specificity(positions: List[str]) -> float:
        num_true_negative: int = positions.count("TN")
        num_false_positive: int = positions.count("FP")
        num_false_negative: int = positions.count("FN")

        if not num_true_negative and not num_false_positive and \
                not num_false_negative:
            return 1.0
        if not num_true_negative and not num_false_positive:
            return 0.0
        else:
            return num_true_negative / (num_true_negative + num_false_positive)

    @staticmethod
    def _calculate_f1(positions: List[str]) -> float:
        precision: float = FeatureImportanceEvaluator._calculate_precision(
            positions)
        recall: float = FeatureImportanceEvaluator._calculate_recall(positions)

        if not precision and not recall:
            return 0.0
        else:
            return 2.0 * ((precision * recall) / (precision + recall))

    def _prepare_r_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df["precision"] = 0.0
        df["recall"] = 0.0
        df["specificity"] = 0.0
        df["f1"] = 0.0
        df["n"] = 0

        for example_id in set(df.example.tolist()):
            example_df = df.loc[df.example == example_id]
            conf_matrix: List[str] = example_df.group.tolist()
            df.loc[df.example == example_id, "precision"] = \
                FeatureImportanceEvaluator._calculate_precision(conf_matrix)
            df.loc[df.example == example_id, "recall"] = \
                FeatureImportanceEvaluator._calculate_recall(conf_matrix)
            df.loc[df.example == example_id, "specificity"] = \
                FeatureImportanceEvaluator._calculate_specificity(conf_matrix)
            df.loc[df.example == example_id, "f1"] = \
                FeatureImportanceEvaluator._calculate_f1(conf_matrix)
            df.loc[df.example == example_id, "n"] = 1.0 / len(example_df.index)

        df["precision"] = df.groupby("label")["precision"].transform("mean")
        df["recall"] = df.groupby("label")["recall"].transform("mean")
        df["specificity"] = df.groupby(
            "label")["specificity"].transform("mean")
        df["f1"] = df.groupby("label")["f1"].transform("mean")
        df["n"] = round(df.groupby("label")["n"].transform("sum"))

        return df

    def _visualize_grammar_agreement(self, results,
                                     set_name: str = "test") -> None:
        df: pd.DataFrame = self._convert_to_data_frame(results)
        if len(df.index) > 0:
            df.to_csv(self.output_dir + set_name +
                      "-grammar-agreement-thresholded-df.txt",
                      sep="\t", index=False)

            plot_script: str = pkg_resources.resource_filename(
                "seqgra", "evaluator/plotagreement.R")
            temp_file_name: str = self.output_dir + set_name + \
                "-thresholded-temp.txt"
            pdf_file_name: str = self.output_dir + set_name + \
                "-grammar-agreement-thresholded.pdf"
            df: pd.DataFrame = self._prepare_r_data_frame(df)
            df.to_csv(temp_file_name, sep="\t", index=False)
            cmd = ["Rscript", "--vanilla", plot_script, temp_file_name,
                    pdf_file_name, self.evaluator_name]
            try:
                subprocess.call(cmd, universal_newlines=True)
            except subprocess.CalledProcessError as exception:
                logging.warning("failed to create grammar-model-agreement "
                                "plots: %s", exception.output)
            except FileNotFoundError as exception:
                logging.warning("Rscript not on PATH, skipping "
                                "grammar-model-agreement plots")
            os.remove(temp_file_name)
