"""MIT - CSAIL - Gifford Lab - seqgra

PyTorch learners

@author: Konstantin Krismer
"""
from typing import List, Any

import torch

from seqgra.learner.dna import DNAMultiClassClassificationLearner
from seqgra.learner.dna import DNAMultiLabelClassificationLearner
from seqgra.learner.torchdataset import DNAMultiClassDataSet
from seqgra.learner.torchdataset import DNAMultiLabelDataSet
from seqgra.parser.modelparser import ModelParser
from seqgra.learner.torchhelper import TorchHelper


class TorchMultiClassClassificationLearner(
    DNAMultiClassClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        training_dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(x_train, y_train, self.labels, True)
        validation_dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(x_val, y_val, self.labels, True)
        TorchHelper.train_model(self, training_dataset, validation_dataset)

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(x, encode_data = encode)
        return TorchHelper.predict(self, dataset, "softmax")

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(x, y, self.labels, True)
        return TorchHelper.evaluate_model(self, dataset)


class TorchMultiLabelClassificationLearner(
    DNAMultiLabelClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        training_dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(x_train, y_train, self.labels, True)
        validation_dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(x_val, y_val, self.labels, True)
        TorchHelper.train_model(self, training_dataset, validation_dataset)

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(x, encode_data = encode)
        return TorchHelper.predict(self, dataset, "sigmoid")

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(x, y, self.labels, True)
        return TorchHelper.evaluate_model(self, dataset)