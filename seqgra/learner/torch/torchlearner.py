"""MIT - CSAIL - Gifford Lab - seqgra

PyTorch learners

@author: Konstantin Krismer
"""
from typing import List, Any

from seqgra.learner import DNAMultiClassClassificationLearner
from seqgra.learner import DNAMultiLabelClassificationLearner
from seqgra.learner import ProteinMultiClassClassificationLearner
from seqgra.learner import ProteinMultiLabelClassificationLearner
from seqgra.learner.torch.torchdataset import DNAMultiClassDataSet
from seqgra.learner.torch.torchdataset import DNAMultiLabelDataSet
from seqgra.learner.torch.torchdataset import ProteinMultiClassDataSet
from seqgra.learner.torch.torchdataset import ProteinMultiLabelDataSet
from seqgra.learner.torch import TorchHelper
from seqgra.model import ModelDefinition


class TorchDNAMultiClassClassificationLearner(
        DNAMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        training_dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(
            x_train, y_train, self.definition.labels, True)
        validation_dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(
            x_val, y_val, self.definition.labels, True)
        TorchHelper.train_model(self, training_dataset, validation_dataset)

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(
            x, encode_data=encode)
        return TorchHelper.predict(self, dataset, "softmax")

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(
            x, y, self.definition.labels, True)
        return TorchHelper.evaluate_model(self, dataset)


class TorchDNAMultiLabelClassificationLearner(
        DNAMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        training_dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(
            x_train, y_train, self.definition.labels, True)
        validation_dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(
            x_val, y_val, self.definition.labels, True)
        TorchHelper.train_model(self, training_dataset, validation_dataset)

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(
            x, encode_data=encode)
        return TorchHelper.predict(self, dataset, "sigmoid")

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(
            x, y, self.definition.labels, True)
        return TorchHelper.evaluate_model(self, dataset)


class TorchProteinMultiClassClassificationLearner(
        ProteinMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        training_dataset: ProteinMultiClassDataSet = ProteinMultiClassDataSet(
            x_train, y_train, self.definition.labels, True)
        validation_dataset: ProteinMultiClassDataSet = ProteinMultiClassDataSet(
            x_val, y_val, self.definition.labels, True)
        TorchHelper.train_model(self, training_dataset, validation_dataset)

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        dataset: ProteinMultiClassDataSet = ProteinMultiClassDataSet(
            x, encode_data=encode)
        return TorchHelper.predict(self, dataset, "softmax")

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        dataset: ProteinMultiClassDataSet = ProteinMultiClassDataSet(
            x, y, self.definition.labels, True)
        return TorchHelper.evaluate_model(self, dataset)


class TorchProteinMultiLabelClassificationLearner(
        ProteinMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        training_dataset: ProteinMultiLabelDataSet = ProteinMultiLabelDataSet(
            x_train, y_train, self.definition.labels, True)
        validation_dataset: ProteinMultiLabelDataSet = ProteinMultiLabelDataSet(
            x_val, y_val, self.definition.labels, True)
        TorchHelper.train_model(self, training_dataset, validation_dataset)

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        dataset: ProteinMultiLabelDataSet = ProteinMultiLabelDataSet(
            x, encode_data=encode)
        return TorchHelper.predict(self, dataset, "sigmoid")

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        dataset: ProteinMultiLabelDataSet = ProteinMultiLabelDataSet(
            x, y, self.definition.labels, True)
        return TorchHelper.evaluate_model(self, dataset)