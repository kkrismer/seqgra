"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod

from seqgra.learner.learner import Learner
from seqgra.mischelper import MiscHelper


class Evaluator(ABC):
    @abstractmethod
    def __init__(self, id: str, learner: Learner, output_dir: str) -> None:
        self.id: str = id
        self.learner: Learner = learner
        self.output_dir = MiscHelper.prepare_path(output_dir + "/" + self.id,
                                                  allow_exists=False)

    @abstractmethod
    def evaluate_model(self, set_name: str = "test") -> None:
        pass
