"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod

from seqgra import MiscHelper
from seqgra.learner import Learner


class Evaluator(ABC):
    @abstractmethod
    def __init__(self, evaluator_id: str, learner: Learner, output_dir: str) -> None:
        self.evaluator_id: str = evaluator_id
        self.learner: Learner = learner
        self.output_dir = MiscHelper.prepare_path(output_dir + "/" +
                                                  self.evaluator_id,
                                                  allow_exists=False)

    @abstractmethod
    def evaluate_model(self, set_name: str = "test") -> None:
        pass
