"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod
import logging
import random
from typing import Any, List, Optional, Set, Tuple

import numpy as np

from seqgra import MiscHelper
from seqgra.learner import Learner


class Evaluator(ABC):
    @abstractmethod
    def __init__(self, evaluator_id: str, learner: Learner,
                 output_dir: str) -> None:
        self.evaluator_id: str = evaluator_id
        self.learner: Learner = learner
        self.output_dir = MiscHelper.prepare_path(output_dir + "/" +
                                                  self.evaluator_id,
                                                  allow_exists=False)

    def evaluate_model(self, set_name: str = "test",
                       subset_idx: Optional[List[int]] = None) -> None:
        x, y, annotations = self._load_data(set_name, subset_idx)
        results = self._evaluate_model(x, y, annotations)
        self._save_results(results, set_name)

    def _load_data(self, set_name: str = "test",
                   subset_idx: Optional[List[int]] = None) -> \
            Tuple[List[str], List[str], List[str]]:
        # load data
        examples_file: str = self.learner.get_examples_file(set_name)
        annotations_file: str = self.learner.get_annotations_file(set_name)
        x, y = self.learner.parse_examples_data(examples_file)
        annotations, _ = self.learner.parse_annotations_data(annotations_file)

        if subset_idx is None:
            return (x, y, annotations)
        else:
            return self.__subset(subset_idx, x, y, annotations)

    @abstractmethod
    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        pass

    @abstractmethod
    def _save_results(self, results, set_name: str = "test") -> None:
        pass

    @staticmethod
    def __subset(idx: List[int], x: List[str], y: List[str],
                 annotations: List[str]) -> Tuple[List[str], List[str], List[str]]:
        if len(x) != len(y) or len(x) != len(annotations):
            raise Exception("x, y, and annotations have to be the same length")

        if max(idx) >= len(x):
            raise Exception("max(idx) >= len(x)")

        x = [x[i] for i in idx]
        y = [y[i] for i in idx]
        annotations = [annotations[i] for i in idx]

        return (x, y, annotations)

    def select_examples(self, threshold: float, set_name: str = "test",
                        labels: Optional[Set[str]] = None) -> Tuple[List[str], List[str], List[str]]:
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
            x, y, annotations = self.__subset(subset_idx, x, y, annotations)

        # predict with learner
        encoded_y = self.learner.encode_y(y)
        y_hat = self.learner.predict(x)

        # discard misclassified / mislabeled examples and
        # examples below threshold
        subset_idx = [i
                      for i in range(len(encoded_y))
                      if np.argmax(y_hat[i]) == np.argmax(encoded_y[i]) and
                      np.max(y_hat[i]) > threshold]
        x, y, annotations = self.__subset(subset_idx, x, y, annotations)

        return (x, y, annotations)

    def select_n_examples(self, n: int, threshold: float,
                          set_name: str = "test",
                          labels: Optional[Set[str]] = None,
                          shuffle: bool = True) -> Tuple[List[str], List[str], List[str]]:
        x, y, annotations = self.select_examples(
            threshold, set_name, labels)

        if len(x) == 0:
            if labels is None:
                logging.warn("no correctly labeled example with prediction "
                             "threshold > " + str(threshold) +
                             " in set '" + set_name + "'")
            else:
                logging.warn("no correctly labeled example with prediction "
                             "threshold > " + str(threshold) +
                             " in set '" + set_name + "' that has one of the"
                             "following labels: " + labels)
            return (None, None, None)
        elif n > len(x):
            if labels is None:
                logging.warn("n (" + str(n) + ") is larger than the number "
                             "of correctly labeled examples with prediction "
                             "threshold > " + str(threshold) +
                             " in set '" + set_name + "' (" + str(len(x)) +
                             "): reset n to " + str(len(x)))
            else:
                logging.warn("n (" + str(n) + ") is larger than the number "
                             "of correctly labeled examples with prediction "
                             "threshold > " + str(threshold) +
                             " in set '" + set_name + "' (" + str(len(x)) +
                             ") for the labels specified: reset n to " +
                             str(len(x)))
            n = len(x)

        if shuffle:
            subset_idx: List[int] = list(range(len(x)))
            random.shuffle(subset_idx)
            subset_idx = subset_idx[:n]
        else:
            subset_idx: List[int] = list(range(n))

        return self.__subset(subset_idx, x, y, annotations)
