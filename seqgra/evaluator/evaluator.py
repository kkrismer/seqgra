"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod
import logging
import random
from typing import Any, List, Optional, Set

import numpy as np

import seqgra.constants as c
from seqgra import AnnotatedExampleSet
from seqgra import MiscHelper
from seqgra.learner import Learner

class Evaluator(ABC):
    @abstractmethod
    def __init__(self, evaluator_id: str, learner: Learner,
                 output_dir: str, threshold: float = 0.5,
                 create_plots: bool = True,
                 supported_tasks: Optional[Set[str]] = None,
                 supported_sequence_spaces: Optional[Set[str]] = None,
                 supported_libraries: Optional[Set[str]] = None) -> None:
        self.evaluator_id: str = evaluator_id
        self.learner: Learner = learner
        self.output_dir = MiscHelper.prepare_path(output_dir + "/" +
                                                  self.evaluator_id,
                                                  allow_exists=False)
        self.threshold: float = threshold
        self.create_plots: bool = create_plots
        if supported_tasks is None:
            self.supported_tasks: Set[str] = c.TaskType.ALL_TASKS
        else:
            self.supported_tasks: Set[str] = set(supported_tasks)
        if supported_sequence_spaces is None:
            self.supported_sequence_spaces: Set[str] = c.SequenceSpaceType.ALL_SEQUENCE_SPACES
        else:
            self.supported_sequence_spaces: Set[str] = set(
                supported_sequence_spaces)
        if supported_libraries is None:
            self.supported_libraries: Set[str] = c.LibraryType.ALL_LIBRARIES
        else:
            self.supported_libraries: Set[str] = set(supported_libraries)
        self.__detect_incompatibilities()

    def __detect_incompatibilities(self):
        if not self.learner.definition.task in self.supported_tasks:
            logging.warning("learner task incompatible with evaluator")
        if not self.learner.definition.sequence_space in self.supported_sequence_spaces:
            logging.warning("learner sequence space incompatible with "
                            "evaluator")
        if not self.learner.definition.library in self.supported_libraries:
            logging.warning("learner library incompatible with evaluator")

    def evaluate_model(self, set_name: str = "test",
                       subset_idx: Optional[List[int]] = None) -> None:
        # TODO get from outside
        subset_idx = [0, 1, 2, 4, 5]
        x, y, annotations = self._load_data(set_name, subset_idx)
        results = self._evaluate_model(x, y, annotations)
        self._save_results(results, set_name)

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
    def _save_results(self, results, set_name: str = "test") -> None:
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

    def select_examples(self, set_name: str = "test",
                        labels: Optional[Set[str]] = None) -> AnnotatedExampleSet:
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
            subset_idx = [i
                          for i, label in enumerate(y)
                          if label in labels]
            x, y, annotations = self._subset(subset_idx, x, y, annotations)

        # predict with learner
        encoded_y = self.learner.encode_y(y)
        y_hat = self.learner.predict(x)

        # discard misclassified / mislabeled examples and
        # examples below threshold
        # TODO fix: i should be label index in model
        subset_idx = [i
                      for i in range(len(encoded_y))
                      if np.argmax(y_hat[i]) == np.argmax(encoded_y[i]) and
                      np.max(y_hat[i]) > self.threshold]
        x, y, annotations = self._subset(subset_idx, x, y, annotations)

        return AnnotatedExampleSet(x, y, annotations)

    def select_n_examples(self, n: int,
                          set_name: str = "test",
                          labels: Optional[Set[str]] = None,
                          per_label: bool = True,
                          shuffle: bool = True) -> AnnotatedExampleSet:
        if labels is not None and per_label:
            x: List[str] = list()
            y: List[str] = list()
            annotations: List[str] = list()
            for label in labels:
                examples = self.select_n_examples(n, set_name, label,
                                                  shuffle)
                x += examples.x
                y += examples.y
                annotations += examples.annotations
            return AnnotatedExampleSet(x, y, annotations)
        else:
            x, y, annotations = self.select_examples(set_name, labels)

            if not x:
                if labels is None:
                    logging.warning("no correctly labeled example with "
                                    "prediction threshold > %s in set '%s'",
                                    self.threshold, set_name)
                else:
                    logging.warning("no correctly labeled example with "
                                    "prediction threshold > %s in set '%s' "
                                    "that has one of the following labels: %s",
                                    self.threshold, set_name, labels)
                return AnnotatedExampleSet(None, None, None)
            elif n > len(x):
                if labels is None:
                    logging.warning("n (%s) is larger than the number "
                                    "of correctly labeled examples with "
                                    "prediction threshold > %s in set '%s' "
                                    "(%s): reset n to %s",
                                    n, self.threshold, set_name,
                                    len(x), len(x))
                else:
                    logging.warning("n (%s) is larger than the number "
                                    "of correctly labeled examples with "
                                    "prediction threshold > %s in set '%s' "
                                    "(%s) for the labels specified: reset n "
                                    "to %s", n, self.threshold, set_name,
                                    len(x), len(x))
                n = len(x)

            if shuffle:
                subset_idx: List[int] = list(range(len(x)))
                random.shuffle(subset_idx)
                subset_idx = subset_idx[:n]
            else:
                subset_idx: List[int] = list(range(n))

            return self._subset(subset_idx, x, y, annotations)
