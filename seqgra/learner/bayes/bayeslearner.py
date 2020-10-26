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
                 output_dir: str, validate_data: bool = True,
                 silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data,
                         silent=silent)

    def create_model(self) -> None:
        BayesOptimalHelper.create_model(self)

    def print_model_summary(self):
        BayesOptimalHelper.print_model_summary(self)

    def set_seed(self) -> None:
        BayesOptimalHelper.set_seed(self)

    def _train_model(self,
                     training_file: Optional[str] = None,
                     validation_file: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        BayesOptimalHelper.train_model(self)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        # TODO file_name option
        return BayesOptimalHelper.predict(self, x, encode, self.silent)

    def get_num_params(self) -> ModelSize:
        return 0

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        # TODO file_name option
        return BayesOptimalHelper.evaluate_model(self, x, y)


class BayesOptimalDNAMultiLabelClassificationLearner(
        DNAMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True,
                 silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data,
                         silent=silent)

    def create_model(self) -> None:
        BayesOptimalHelper.create_model(self)

    def print_model_summary(self):
        BayesOptimalHelper.print_model_summary(self)

    def set_seed(self) -> None:
        BayesOptimalHelper.set_seed(self)

    def _train_model(self,
                     training_file: Optional[str] = None,
                     validation_file: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        BayesOptimalHelper.train_model(self)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        # TODO file_name option
        return BayesOptimalHelper.predict(self, x, encode, self.silent)

    def get_num_params(self) -> ModelSize:
        return 0

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        # TODO file_name option
        return BayesOptimalHelper.evaluate_model(self, x, y)


class BayesOptimalProteinMultiClassClassificationLearner(
        ProteinMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True,
                 silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data,
                         silent=silent)

    def create_model(self) -> None:
        BayesOptimalHelper.create_model(self)

    def print_model_summary(self):
        BayesOptimalHelper.print_model_summary(self)

    def set_seed(self) -> None:
        BayesOptimalHelper.set_seed(self)

    def _train_model(self,
                     training_file: Optional[str] = None,
                     validation_file: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        BayesOptimalHelper.train_model(self)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        # TODO file_name option
        return BayesOptimalHelper.predict(self, x, encode, self.silent)

    def get_num_params(self) -> ModelSize:
        return 0

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        # TODO file_name option
        return BayesOptimalHelper.evaluate_model(self, x, y)


class BayesOptimalProteinMultiLabelClassificationLearner(
        ProteinMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True,
                 silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data,
                         silent=silent)

    def create_model(self) -> None:
        BayesOptimalHelper.create_model(self)

    def print_model_summary(self):
        BayesOptimalHelper.print_model_summary(self)

    def set_seed(self) -> None:
        BayesOptimalHelper.set_seed(self)

    def _train_model(self,
                     training_file: Optional[str] = None,
                     validation_file: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        BayesOptimalHelper.train_model(self)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        # TODO file_name option
        return BayesOptimalHelper.predict(self, x, encode, self.silent)

    def get_num_params(self) -> ModelSize:
        return 0

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        # TODO file_name option
        return BayesOptimalHelper.evaluate_model(self, x, y)
