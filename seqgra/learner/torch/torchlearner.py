"""MIT - CSAIL - Gifford Lab - seqgra

PyTorch learners

@author: Konstantin Krismer
"""
import logging
from typing import Any, List, Optional

import numpy as np

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
        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in TorchHelper.MULTI_CLASS_CLASSIFICATION_LOSSES:
                logging.warning("loss function '%s' is incompatible with "
                                "multi-class classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if loss == "crossentropyloss":
                return "softmax"
            elif loss == "bcewithlogitsloss":
                logging.warning("activation function 'sigmoid' is "
                                "incompatible with multi-class "
                                "classification models")
                return "sigmoid"
        return None

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        # one hot encode input and labels
        encoded_x_train = self.encode_x(x_train)
        encoded_y_train = self.encode_y(y_train)
        encoded_x_val = self.encode_x(x_val)
        encoded_y_val = self.encode_y(y_val)

        training_dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(
            encoded_x_train, encoded_y_train, self.definition.labels)
        validation_dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(
            encoded_x_val, encoded_y_val, self.definition.labels)

        TorchHelper.train_model(self, training_dataset,
                                validation_dataset,
                                self._get_output_layer_activation_function())

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        if encode:
            x = self.encode_x(x)

        dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(
            x, labels=self.definition.labels)

        return TorchHelper.predict(self, dataset,
                                self._get_output_layer_activation_function())

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        encoded_x = self.encode_x(x)
        encoded_y = self.encode_y(y)

        dataset: DNAMultiClassDataSet = DNAMultiClassDataSet(
            encoded_x, encoded_y, self.definition.labels)

        return TorchHelper.evaluate_model(self, dataset,
                                self._get_output_layer_activation_function())

    def encode_x(self, x: List[str]):
        encoded_x = super().encode_x(x)

        if self.definition.input_encoding == "2D":
            # from (N, W, C) to (H, N, W, C)
            encoded_x = np.expand_dims(encoded_x, axis=0)
            # from (H, N, W, C) to (N, C, H, W)
            encoded_x = np.transpose(encoded_x, (1, 3, 0, 2))
        else:
            # from (N, W, C) to (N, C, W)
            encoded_x = np.transpose(encoded_x, (0, 2, 1))

        return encoded_x

    def decode_x(self, x):
        if self.definition.input_encoding == "2D":
            # from (N, C, H, W) to (N, C, W)
            x = np.squeeze(x, axis=2)

        # from (N, C, W) to (N, W, C)
        x = np.transpose(x, (0, 2, 1))

        return super().decode_x(x)


class TorchDNAMultiLabelClassificationLearner(
        DNAMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)
        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in TorchHelper.MULTI_LABEL_CLASSIFICATION_LOSSES:
                logging.warning("loss function '%s' is incompatible with "
                                "multi-label classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if loss == "crossentropyloss":
                logging.warning("activation function 'softmax' is "
                                "incompatible with multi-label "
                                "classification models")
                return "softmax"
            elif loss == "bcewithlogitsloss":
                return "sigmoid"
        return None

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        # one hot encode input and labels
        encoded_x_train = self.encode_x(x_train)
        encoded_y_train = self.encode_y(y_train)
        encoded_x_val = self.encode_x(x_val)
        encoded_y_val = self.encode_y(y_val)

        training_dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(
            encoded_x_train, encoded_y_train, self.definition.labels)
        validation_dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(
            encoded_x_val, encoded_y_val, self.definition.labels)

        TorchHelper.train_model(self, training_dataset,
                                validation_dataset,
                                self._get_output_layer_activation_function())

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        if encode:
            x = self.encode_x(x)

        dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(
            x, labels=self.definition.labels)

        return TorchHelper.predict(self, dataset,
                                self._get_output_layer_activation_function())

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        encoded_x = self.encode_x(x)
        encoded_y = self.encode_y(y)

        dataset: DNAMultiLabelDataSet = DNAMultiLabelDataSet(
            encoded_x, encoded_y, self.definition.labels)

        return TorchHelper.evaluate_model(self, dataset,
                                self._get_output_layer_activation_function())

    def encode_x(self, x: List[str]):
        encoded_x = super().encode_x(x)

        if self.definition.input_encoding == "2D":
            # from (N, W, C) to (H, N, W, C)
            encoded_x = np.expand_dims(encoded_x, axis=0)
            # from (H, N, W, C) to (N, C, H, W)
            encoded_x = np.transpose(encoded_x, (1, 3, 0, 2))
        else:
            # from (N, W, C) to (N, C, W)
            encoded_x = np.transpose(encoded_x, (0, 2, 1))

        return encoded_x

    def decode_x(self, x):
        if self.definition.input_encoding == "2D":
            # from (N, C, H, W) to (N, C, W)
            x = np.squeeze(x, axis=2)

        # from (N, C, W) to (N, W, C)
        x = np.transpose(x, (0, 2, 1))

        return super().decode_x(x)


class TorchProteinMultiClassClassificationLearner(
        ProteinMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)
        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in TorchHelper.MULTI_CLASS_CLASSIFICATION_LOSSES:
                logging.warning("loss function '%s' is incompatible with "
                                "multi-class classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if loss == "crossentropyloss":
                return "softmax"
            elif loss == "bcewithlogitsloss":
                logging.warning("activation function 'sigmoid' is "
                                "incompatible with multi-class "
                                "classification models")
                return "sigmoid"
        return None

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        # one hot encode input and labels
        encoded_x_train = self.encode_x(x_train)
        encoded_y_train = self.encode_y(y_train)
        encoded_x_val = self.encode_x(x_val)
        encoded_y_val = self.encode_y(y_val)

        training_dataset: ProteinMultiClassDataSet = ProteinMultiClassDataSet(
            encoded_x_train, encoded_y_train, self.definition.labels)
        validation_dataset: ProteinMultiClassDataSet = ProteinMultiClassDataSet(
            encoded_x_val, encoded_y_val, self.definition.labels)

        TorchHelper.train_model(self, training_dataset,
                                validation_dataset,
                                self._get_output_layer_activation_function())

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        if encode:
            x = self.encode_x(x)

        dataset: ProteinMultiClassDataSet = ProteinMultiClassDataSet(
            x, labels=self.definition.labels)

        return TorchHelper.predict(self, dataset,
                                self._get_output_layer_activation_function())

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        encoded_x = self.encode_x(x)
        encoded_y = self.encode_y(y)

        dataset: ProteinMultiClassDataSet = ProteinMultiClassDataSet(
            encoded_x, encoded_y, self.definition.labels)

        return TorchHelper.evaluate_model(self, dataset,
                                self._get_output_layer_activation_function())

    def encode_x(self, x: List[str]):
        encoded_x = super().encode_x(x)

        if self.definition.input_encoding == "2D":
            # from (N, W, C) to (H, N, W, C)
            encoded_x = np.expand_dims(encoded_x, axis=0)
            # from (H, N, W, C) to (N, C, H, W)
            encoded_x = np.transpose(encoded_x, (1, 3, 0, 2))
        else:
            # from (N, W, C) to (N, C, W)
            encoded_x = np.transpose(encoded_x, (0, 2, 1))

        return encoded_x

    def decode_x(self, x):
        if self.definition.input_encoding == "2D":
            # from (N, C, H, W) to (N, C, W)
            x = np.squeeze(x, axis=2)

        # from (N, C, W) to (N, W, C)
        x = np.transpose(x, (0, 2, 1))

        return super().decode_x(x)


class TorchProteinMultiLabelClassificationLearner(
        ProteinMultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)
        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in TorchHelper.MULTI_LABEL_CLASSIFICATION_LOSSES:
                logging.warning("loss function '%s' is incompatible with "
                                "multi-label classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if loss == "crossentropyloss":
                logging.warning("activation function 'softmax' is "
                                "incompatible with multi-label "
                                "classification models")
                return "softmax"
            elif loss == "bcewithlogitsloss":
                return "sigmoid"
        return None

    def create_model(self) -> None:
        TorchHelper.create_model(self)

    def print_model_summary(self):
        TorchHelper.print_model_summary(self)

    def set_seed(self) -> None:
        TorchHelper.set_seed(self)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        # one hot encode input and labels
        encoded_x_train = self.encode_x(x_train)
        encoded_y_train = self.encode_y(y_train)
        encoded_x_val = self.encode_x(x_val)
        encoded_y_val = self.encode_y(y_val)

        training_dataset: ProteinMultiLabelDataSet = ProteinMultiLabelDataSet(
            encoded_x_train, encoded_y_train, self.definition.labels)
        validation_dataset: ProteinMultiLabelDataSet = ProteinMultiLabelDataSet(
            encoded_x_val, encoded_y_val, self.definition.labels)

        TorchHelper.train_model(self, training_dataset,
                                validation_dataset,
                                self._get_output_layer_activation_function())

    def save_model(self, model_name: str = "") -> None:
        TorchHelper.save_model(self, model_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, model_name: str = "") -> None:
        TorchHelper.load_model(self, model_name)

    def predict(self, x: Any, encode: bool = True):
        if encode:
            x = self.encode_x(x)

        dataset: ProteinMultiLabelDataSet = ProteinMultiLabelDataSet(
            x, labels=self.definition.labels)

        return TorchHelper.predict(self, dataset,
                                self._get_output_layer_activation_function())

    def get_num_params(self):
        return TorchHelper.get_num_params(self)

    def _evaluate_model(self, x: List[str], y: List[str]):
        encoded_x = self.encode_x(x)
        encoded_y = self.encode_y(y)

        dataset: ProteinMultiLabelDataSet = ProteinMultiLabelDataSet(
            encoded_x, encoded_y, self.definition.labels)

        return TorchHelper.evaluate_model(self, dataset,
                                self._get_output_layer_activation_function())

    def encode_x(self, x: List[str]):
        encoded_x = super().encode_x(x)

        if self.definition.input_encoding == "2D":
            # from (N, W, C) to (H, N, W, C)
            encoded_x = np.expand_dims(encoded_x, axis=0)
            # from (H, N, W, C) to (N, C, H, W)
            encoded_x = np.transpose(encoded_x, (1, 3, 0, 2))
        else:
            # from (N, W, C) to (N, C, W)
            encoded_x = np.transpose(encoded_x, (0, 2, 1))

        return encoded_x

    def decode_x(self, x):
        if self.definition.input_encoding == "2D":
            # from (N, C, H, W) to (N, C, W)
            x = np.squeeze(x, axis=2)

        # from (N, C, W) to (N, W, C)
        x = np.transpose(x, (0, 2, 1))

        return super().decode_x(x)
