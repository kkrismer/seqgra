"""MIT - CSAIL - Gifford Lab - seqgra

TensorFlow Keras learners

@author: Konstantin Krismer
"""
from typing import Any, List, Optional

from seqgra import ModelSize
from seqgra.learner import DNAMultiClassClassificationLearner
from seqgra.learner import DNAMultiLabelClassificationLearner
from seqgra.learner import ProteinMultiClassClassificationLearner
from seqgra.learner import ProteinMultiLabelClassificationLearner
from seqgra.learner.bayes import BayesOptimalHelper
from seqgra.model import ModelDefinition


class BayesOptimalDNAMultiClassClassificationLearner(
        DNAMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data)

    def create_model(self) -> None:
        BayesOptimalHelper.create_model(self)

    def print_model_summary(self):
        BayesOptimalHelper.print_model_summary(self)

    def set_seed(self) -> None:
        BayesOptimalHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        BayesOptimalHelper.train_model(self)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def predict(self, x: Any, encode: bool = True):
        return BayesOptimalHelper.predict(self, x, encode)

    def get_num_params(self) -> ModelSize:
        return 0

    def _evaluate_model(self, x: List[str], y: List[str]):
        return BayesOptimalHelper.evaluate_model(self, x, y)


class BayesOptimalDNAMultiLabelClassificationLearner(
        DNAMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data)

    def create_model(self) -> None:
        BayesOptimalHelper.create_model(self)

    def print_model_summary(self):
        BayesOptimalHelper.print_model_summary(self)

    def set_seed(self) -> None:
        BayesOptimalHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        BayesOptimalHelper.train_model(self)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def predict(self, x: Any, encode: bool = True):
        return BayesOptimalHelper.predict(self, x, encode)

    def get_num_params(self) -> ModelSize:
        return 0

    def _evaluate_model(self, x: List[str], y: List[str]):
        return BayesOptimalHelper.evaluate_model(self, x, y)


class BayesOptimalProteinMultiClassClassificationLearner(
        ProteinMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data)

    def create_model(self) -> None:
        BayesOptimalHelper.create_model(self)

    def print_model_summary(self):
        BayesOptimalHelper.print_model_summary(self)

    def set_seed(self) -> None:
        BayesOptimalHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        BayesOptimalHelper.train_model(self)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def predict(self, x: Any, encode: bool = True):
        return BayesOptimalHelper.predict(self, x, encode)

    def get_num_params(self) -> ModelSize:
        return 0

    def _evaluate_model(self, x: List[str], y: List[str]):
        return BayesOptimalHelper.evaluate_model(self, x, y)


class BayesOptimalProteinMultiLabelClassificationLearner(
        ProteinMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data)

    def create_model(self) -> None:
        BayesOptimalHelper.create_model(self)

    def print_model_summary(self):
        BayesOptimalHelper.print_model_summary(self)

    def set_seed(self) -> None:
        BayesOptimalHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        BayesOptimalHelper.train_model(self)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def predict(self, x: Any, encode: bool = True):
        return BayesOptimalHelper.predict(self, x, encode)

    def get_num_params(self) -> ModelSize:
        return 0

    def _evaluate_model(self, x: List[str], y: List[str]):
        return BayesOptimalHelper.evaluate_model(self, x, y)
