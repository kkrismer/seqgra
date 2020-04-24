"""MIT - CSAIL - Gifford Lab - seqgra

TensorFlow Keras learners

@author: Konstantin Krismer
"""
from typing import List, Any

from seqgra.learner import DNAMultiClassClassificationLearner
from seqgra.learner import DNAMultiLabelClassificationLearner
from seqgra.learner import ProteinMultiClassClassificationLearner
from seqgra.learner import ProteinMultiLabelClassificationLearner
from seqgra.learner.tensorflow import KerasHelper
from seqgra.model import ModelDefinition


class KerasDNAMultiClassClassificationLearner(
        DNAMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

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


class KerasDNAMultiLabelClassificationLearner(
        DNAMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

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


class KerasProteinMultiClassClassificationLearner(
        ProteinMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

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


class KerasProteinMultiLabelClassificationLearner(
        ProteinMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

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
