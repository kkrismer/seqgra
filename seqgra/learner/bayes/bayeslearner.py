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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        BayesOptimalHelper.train_model(self)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            pass
        elif file_name is not None:
            x, y = self.parse_examples_data(file_name)
        else:
            raise Exception("specify either file_name or x, y")

        x = self.encode_x(x)
        y = self.encode_y(y)
        return BayesOptimalHelper.evaluate_model(self, x, y)

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
        elif file_name is not None:
            x, _ = self.parse_examples_data(file_name)
            x = self.encode_x(x)
        else:
            raise Exception("specify either file_name or x")

        return BayesOptimalHelper.predict(self, x, self.silent)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def get_num_params(self) -> ModelSize:
        return 0


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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        BayesOptimalHelper.train_model(self)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            pass
        elif file_name is not None:
            x, y = self.parse_examples_data(file_name)
        else:
            raise Exception("specify either file_name or x, y")

        x = self.encode_x(x)
        y = self.encode_y(y)
        return BayesOptimalHelper.evaluate_model(self, x, y)

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
        elif file_name is not None:
            x, _ = self.parse_examples_data(file_name)
            x = self.encode_x(x)
        else:
            raise Exception("specify either file_name or x")

        return BayesOptimalHelper.predict(self, x, self.silent)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def get_num_params(self) -> ModelSize:
        return 0


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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        BayesOptimalHelper.train_model(self)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            pass
        elif file_name is not None:
            x, y = self.parse_examples_data(file_name)
        else:
            raise Exception("specify either file_name or x, y")

        x = self.encode_x(x)
        y = self.encode_y(y)
        return BayesOptimalHelper.evaluate_model(self, x, y)

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
        elif file_name is not None:
            x, _ = self.parse_examples_data(file_name)
            x = self.encode_x(x)
        else:
            raise Exception("specify either file_name or x")

        return BayesOptimalHelper.predict(self, x, self.silent)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def get_num_params(self) -> ModelSize:
        return 0


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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        BayesOptimalHelper.train_model(self)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            pass
        elif file_name is not None:
            x, y = self.parse_examples_data(file_name)
        else:
            raise Exception("specify either file_name or x, y")

        x = self.encode_x(x)
        y = self.encode_y(y)
        return BayesOptimalHelper.evaluate_model(self, x, y)

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
        elif file_name is not None:
            x, _ = self.parse_examples_data(file_name)
            x = self.encode_x(x)
        else:
            raise Exception("specify either file_name or x")

        return BayesOptimalHelper.predict(self, x, self.silent)

    def save_model(self, file_name: Optional[str] = None):
        pass

    def write_session_info(self) -> None:
        BayesOptimalHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None):
        self.create_model()

    def get_num_params(self) -> ModelSize:
        return 0
