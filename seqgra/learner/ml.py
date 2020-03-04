"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from typing import List, Any

from seqgra.learner.dna import DNAMultiLabelClassificationLearner
from seqgra.parser.modelparser import ModelParser
from seqgra.learner.kerashelper import KerasHelper


class KerasSequentialMultiLabelClassificationLearner(DNAMultiLabelClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

    def create_model(self) -> None:
        KerasHelper.create_model(self)

    def print_model_summary(self):
        KerasHelper.print_model_summary(self)

    def set_seed(self) -> None:
        KerasHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        KerasHelper.train_model(self, x_train, y_train, x_val, y_val)

    def save_model(self, model_name: str = "") -> None:
        KerasHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        KerasHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        KerasHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        return KerasHelper.predict(self, x, encode)

    def get_num_params(self):
        return KerasHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        return KerasHelper.evaluate_model(self, x, y)
