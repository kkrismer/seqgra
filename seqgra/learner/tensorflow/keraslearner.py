"""MIT - CSAIL - Gifford Lab - seqgra

TensorFlow Keras learners

@author: Konstantin Krismer
"""
from distutils.util import strtobool
from typing import Any, List, Optional

import numpy as np

from seqgra.learner import DNAMultiClassClassificationLearner
from seqgra.learner import DNAMultiLabelClassificationLearner
from seqgra.learner import ProteinMultiClassClassificationLearner
from seqgra.learner import ProteinMultiLabelClassificationLearner
from seqgra.learner.tensorflow import KerasHelper
from seqgra.model import ModelDefinition


class KerasDNAMultiClassClassificationLearner(
        DNAMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data)
        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in KerasHelper.MULTI_CLASS_CLASSIFICATION_LOSSES:
                self.logger.warning("loss function '%s' is incompatible with "
                                    "multi-class classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "from_logits" in self.definition.loss_hyperparameters and \
                "loss" in self.definition.loss_hyperparameters:
            from_logits: bool = bool(strtobool(
                self.definition.loss_hyperparameters["from_logits"]))
            if from_logits:
                loss: str = self.definition.loss_hyperparameters["loss"]
                loss = loss.lower().replace("_", "").strip()
                if loss == "categoricalcrossentropy" or \
                        loss == "sparsecategoricalcrossentropy":
                    return "softmax"
                elif loss == "binarycrossentropy":
                    self.logger.warning("activation function 'sigmoid' is "
                                        "incompatible with multi-class "
                                        "classification models")
                    return "sigmoid"
        return None

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

    def encode_x(self, x: List[str]):
        encoded_x = super().encode_x(x)

        if self.definition.input_encoding == "2D":
            # from (N, W, C) to (N, H, W, C)
            encoded_x = np.expand_dims(encoded_x, axis=1)

        return encoded_x

    def decode_x(self, x):
        if self.definition.input_encoding == "2D":
            # from (N, H, W, C) to (N, W, C)
            x = np.squeeze(x, axis=1)

        return super().decode_x(x)


class KerasDNAMultiLabelClassificationLearner(
        DNAMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data)
        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in KerasHelper.MULTI_LABEL_CLASSIFICATION_LOSSES:
                self.logger.warning("loss function '%s' is incompatible with "
                                    "multi-label classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "from_logits" in self.definition.loss_hyperparameters and \
                "loss" in self.definition.loss_hyperparameters:
            from_logits: bool = bool(strtobool(
                self.definition.loss_hyperparameters["from_logits"]))
            if from_logits:
                loss: str = self.definition.loss_hyperparameters["loss"]
                loss = loss.lower().replace("_", "").strip()
                if loss == "categoricalcrossentropy" or \
                        loss == "sparsecategoricalcrossentropy":
                    self.logger.warning("activation function 'sofmax' is "
                                        "incompatible with multi-label "
                                        "classification models")
                    return "softmax"
                elif loss == "binarycrossentropy":
                    return "sigmoid"
        return None

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

    def encode_x(self, x: List[str]):
        encoded_x = super().encode_x(x)

        if self.definition.input_encoding == "2D":
            # from (N, W, C) to (N, H, W, C)
            encoded_x = np.expand_dims(encoded_x, axis=1)

        return encoded_x

    def decode_x(self, x):
        if self.definition.input_encoding == "2D":
            # from (N, H, W, C) to (N, W, C)
            x = np.squeeze(x, axis=1)

        return super().decode_x(x)


class KerasProteinMultiClassClassificationLearner(
        ProteinMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data)
        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in KerasHelper.MULTI_CLASS_CLASSIFICATION_LOSSES:
                self.logger.warning("loss function '%s' is incompatible with "
                                    "multi-class classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "from_logits" in self.definition.loss_hyperparameters and \
                "loss" in self.definition.loss_hyperparameters:
            from_logits: bool = bool(strtobool(
                self.definition.loss_hyperparameters["from_logits"]))
            if from_logits:
                loss: str = self.definition.loss_hyperparameters["loss"]
                loss = loss.lower().replace("_", "").strip()
                if loss == "categoricalcrossentropy" or \
                        loss == "sparsecategoricalcrossentropy":
                    return "softmax"
                elif loss == "binarycrossentropy":
                    self.logger.warning("activation function 'sigmoid' is "
                                        "incompatible with multi-class "
                                        "classification models")
                    return "sigmoid"
        return None

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

    def encode_x(self, x: List[str]):
        encoded_x = super().encode_x(x)

        if self.definition.input_encoding == "2D":
            # from (N, W, C) to (N, H, W, C)
            encoded_x = np.expand_dims(encoded_x, axis=1)

        return encoded_x

    def decode_x(self, x):
        if self.definition.input_encoding == "2D":
            # from (N, H, W, C) to (N, W, C)
            x = np.squeeze(x, axis=1)

        return super().decode_x(x)


class KerasProteinMultiLabelClassificationLearner(
        ProteinMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True) -> None:
        super().__init__(model_definition, data_dir, output_dir, validate_data)
        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in KerasHelper.MULTI_LABEL_CLASSIFICATION_LOSSES:
                self.logger.warning("loss function '%s' is incompatible with "
                                    "multi-label classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "from_logits" in self.definition.loss_hyperparameters and \
                "loss" in self.definition.loss_hyperparameters:
            from_logits: bool = bool(strtobool(
                self.definition.loss_hyperparameters["from_logits"]))
            if from_logits:
                loss: str = self.definition.loss_hyperparameters["loss"]
                loss = loss.lower().replace("_", "").strip()
                if loss == "categoricalcrossentropy" or \
                        loss == "sparsecategoricalcrossentropy":
                    self.logger.warning("activation function 'softmax' is "
                                        "incompatible with multi-label "
                                        "classification models")
                    return "softmax"
                elif loss == "binarycrossentropy":
                    return "sigmoid"
        return None

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

    def encode_x(self, x: List[str]):
        encoded_x = super().encode_x(x)

        if self.definition.input_encoding == "2D":
            # from (N, W, C) to (N, H, W, C)
            encoded_x = np.expand_dims(encoded_x, axis=1)

        return encoded_x

    def decode_x(self, x):
        if self.definition.input_encoding == "2D":
            # from (N, H, W, C) to (N, W, C)
            x = np.squeeze(x, axis=1)

        return super().decode_x(x)
