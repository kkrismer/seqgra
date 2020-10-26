"""MIT - CSAIL - Gifford Lab - seqgra

PyTorch learners

@author: Konstantin Krismer
"""
from typing import Any, List, Optional

import numpy as np
import torch

from seqgra import ModelSize
from seqgra.learner import DNAMultiClassClassificationLearner
from seqgra.learner import DNAMultiLabelClassificationLearner
from seqgra.learner import ProteinMultiClassClassificationLearner
from seqgra.learner import ProteinMultiLabelClassificationLearner
from seqgra.learner.torch.torchdataset import MultiClassDataSet
from seqgra.learner.torch.torchdataset import MultiLabelDataSet
from seqgra.learner.torch.torchdataset import IterableMultiClassDataSet
from seqgra.learner.torch.torchdataset import IterableMultiLabelDataSet
from seqgra.learner.torch import TorchHelper
from seqgra.model import ModelDefinition


class TorchDNAMultiClassClassificationLearner(
        DNAMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0, silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id, silent=silent)
        self.use_cuda: bool = torch.cuda.is_available() and gpu_id != -1
        if self.use_cuda:
            self.device_label: str = "cuda:" + str(gpu_id)
        else:
            self.device_label: str = "cpu"
        self.device = torch.device(self.device_label)

        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in TorchHelper.MULTI_CLASS_CLASSIFICATION_LOSSES:
                self.logger.warning("loss function '%s' is incompatible with "
                                    "multi-class classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if loss == "crossentropyloss":
                return "softmax"
            elif loss == "bcewithlogitsloss":
                self.logger.warning("activation function 'sigmoid' is "
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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        if x_train is not None and y_train is not None:
            training_dataset: torch.utils.data.Dataset = MultiClassDataSet(
                self.encode_x(x_train), self.encode_y(y_train))
        elif file_name_train is not None:
            training_dataset: torch.utils.data.Dataset = IterableMultiClassDataSet(
                file_name_train, self)
        else:
            raise Exception(
                "specify either file_name_train or x_train, y_train")

        if x_val is not None and y_val is not None:
            validation_dataset: torch.utils.data.Dataset = MultiClassDataSet(
                self.encode_x(x_val), self.encode_y(y_val))
        elif file_name_val is not None:
            validation_dataset: torch.utils.data.Dataset = IterableMultiClassDataSet(
                file_name_val, self)
        else:
            raise Exception("specify either file_name_val or x_val, y_val")

        TorchHelper.train_model(self, training_dataset, validation_dataset,
                                self._get_output_layer_activation_function(),
                                silent=self.silent)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            dataset: torch.utils.data.Dataset = MultiClassDataSet(
                self.encode_x(x), self.encode_y(y))
        elif file_name is not None:
            dataset: torch.utils.data.Dataset = IterableMultiClassDataSet(
                file_name, self)
        else:
            raise Exception("specify either file_name or x, y")

        return TorchHelper.evaluate_model(
            self, dataset, self._get_output_layer_activation_function())

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
            dataset: torch.utils.data.Dataset = MultiClassDataSet(x)
        elif file_name is not None:
            dataset: torch.utils.data.Dataset = IterableMultiClassDataSet(
                file_name, self, False)
        else:
            raise Exception("specify either file_name or x")

        return TorchHelper.predict(
            self, dataset, self._get_output_layer_activation_function())

    def save_model(self, file_name: Optional[str] = None) -> None:
        TorchHelper.save_model(self, file_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None) -> None:
        TorchHelper.load_model(self, file_name)

    def get_num_params(self) -> ModelSize:
        return TorchHelper.get_num_params(self)

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
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0, silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id, silent=silent)
        self.use_cuda: bool = torch.cuda.is_available() and gpu_id != -1
        if self.use_cuda:
            self.device_label: str = "cuda:" + str(gpu_id)
        else:
            self.device_label: str = "cpu"
        self.device = torch.device(self.device_label)

        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in TorchHelper.MULTI_LABEL_CLASSIFICATION_LOSSES:
                self.logger.warning("loss function '%s' is incompatible with "
                                    "multi-label classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if loss == "crossentropyloss":
                self.logger.warning("activation function 'softmax' is "
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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        if x_train is not None and y_train is not None:
            training_dataset: torch.utils.data.Dataset = MultiLabelDataSet(
                self.encode_x(x_train), self.encode_y(y_train))
        elif file_name_train is not None:
            training_dataset: torch.utils.data.Dataset = IterableMultiLabelDataSet(
                file_name_train, self)
        else:
            raise Exception(
                "specify either file_name_train or x_train, y_train")

        if x_val is not None and y_val is not None:
            validation_dataset: torch.utils.data.Dataset = MultiLabelDataSet(
                self.encode_x(x_val), self.encode_y(y_val))
        elif file_name_val is not None:
            validation_dataset: torch.utils.data.Dataset = IterableMultiLabelDataSet(
                file_name_val, self)
        else:
            raise Exception("specify either file_name_val or x_val, y_val")

        TorchHelper.train_model(self, training_dataset, validation_dataset,
                                self._get_output_layer_activation_function(),
                                silent=self.silent)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            dataset: torch.utils.data.Dataset = MultiLabelDataSet(
                self.encode_x(x), self.encode_y(y))
        elif file_name is not None:
            dataset: torch.utils.data.Dataset = IterableMultiLabelDataSet(
                file_name, self)
        else:
            raise Exception("specify either file_name or x, y")

        return TorchHelper.evaluate_model(
            self, dataset, self._get_output_layer_activation_function())

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
            dataset: torch.utils.data.Dataset = MultiLabelDataSet(x)
        elif file_name is not None:
            dataset: torch.utils.data.Dataset = IterableMultiLabelDataSet(
                file_name, self, False)
        else:
            raise Exception("specify either file_name or x")

        return TorchHelper.predict(
            self, dataset, self._get_output_layer_activation_function())

    def save_model(self, file_name: Optional[str] = None) -> None:
        TorchHelper.save_model(self, file_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None) -> None:
        TorchHelper.load_model(self, file_name)

    def get_num_params(self) -> ModelSize:
        return TorchHelper.get_num_params(self)

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
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0, silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id, silent=silent)
        self.use_cuda: bool = torch.cuda.is_available() and gpu_id != -1
        if self.use_cuda:
            self.device_label: str = "cuda:" + str(gpu_id)
        else:
            self.device_label: str = "cpu"
        self.device = torch.device(self.device_label)

        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in TorchHelper.MULTI_CLASS_CLASSIFICATION_LOSSES:
                self.logger.warning("loss function '%s' is incompatible with "
                                    "multi-class classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if loss == "crossentropyloss":
                return "softmax"
            elif loss == "bcewithlogitsloss":
                self.logger.warning("activation function 'sigmoid' is "
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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        if x_train is not None and y_train is not None:
            training_dataset: torch.utils.data.Dataset = MultiClassDataSet(
                self.encode_x(x_train), self.encode_y(y_train))
        elif file_name_train is not None:
            training_dataset: torch.utils.data.Dataset = IterableMultiClassDataSet(
                file_name_train, self)
        else:
            raise Exception(
                "specify either file_name_train or x_train, y_train")

        if x_val is not None and y_val is not None:
            validation_dataset: torch.utils.data.Dataset = MultiClassDataSet(
                self.encode_x(x_val), self.encode_y(y_val))
        elif file_name_val is not None:
            validation_dataset: torch.utils.data.Dataset = IterableMultiClassDataSet(
                file_name_val, self)
        else:
            raise Exception("specify either file_name_val or x_val, y_val")

        TorchHelper.train_model(self, training_dataset, validation_dataset,
                                self._get_output_layer_activation_function(),
                                silent=self.silent)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            dataset: torch.utils.data.Dataset = MultiClassDataSet(
                self.encode_x(x), self.encode_y(y))
        elif file_name is not None:
            dataset: torch.utils.data.Dataset = IterableMultiClassDataSet(
                file_name, self)
        else:
            raise Exception("specify either file_name or x, y")

        return TorchHelper.evaluate_model(
            self, dataset, self._get_output_layer_activation_function())

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
            dataset: torch.utils.data.Dataset = MultiClassDataSet(x)
        elif file_name is not None:
            dataset: torch.utils.data.Dataset = IterableMultiClassDataSet(
                file_name, self, False)
        else:
            raise Exception("specify either file_name or x")

        return TorchHelper.predict(
            self, dataset, self._get_output_layer_activation_function())

    def save_model(self, file_name: Optional[str] = None) -> None:
        TorchHelper.save_model(self, file_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None) -> None:
        TorchHelper.load_model(self, file_name)

    def get_num_params(self) -> ModelSize:
        return TorchHelper.get_num_params(self)

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
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0, silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id, silent=silent)
        self.use_cuda: bool = torch.cuda.is_available() and gpu_id != -1
        if self.use_cuda:
            self.device_label: str = "cuda:" + str(gpu_id)
        else:
            self.device_label: str = "cpu"
        self.device = torch.device(self.device_label)

        self._check_task_loss_compatibility()

    def _check_task_loss_compatibility(self) -> None:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if not loss in TorchHelper.MULTI_LABEL_CLASSIFICATION_LOSSES:
                self.logger.warning("loss function '%s' is incompatible with "
                                    "multi-label classification models", loss)

    def _get_output_layer_activation_function(self) -> Optional[str]:
        if "loss" in self.definition.loss_hyperparameters:
            loss: str = self.definition.loss_hyperparameters["loss"]
            loss = loss.lower().replace("_", "").strip()
            if loss == "crossentropyloss":
                self.logger.warning("activation function 'softmax' is "
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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        if x_train is not None and y_train is not None:
            training_dataset: torch.utils.data.Dataset = MultiLabelDataSet(
                self.encode_x(x_train), self.encode_y(y_train))
        elif file_name_train is not None:
            training_dataset: torch.utils.data.Dataset = IterableMultiLabelDataSet(
                file_name_train, self)
        else:
            raise Exception(
                "specify either file_name_train or x_train, y_train")

        if x_val is not None and y_val is not None:
            validation_dataset: torch.utils.data.Dataset = MultiLabelDataSet(
                self.encode_x(x_val), self.encode_y(y_val))
        elif file_name_val is not None:
            validation_dataset: torch.utils.data.Dataset = IterableMultiLabelDataSet(
                file_name_val, self)
        else:
            raise Exception("specify either file_name_val or x_val, y_val")

        TorchHelper.train_model(self, training_dataset, validation_dataset,
                                self._get_output_layer_activation_function(),
                                silent=self.silent)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            dataset: torch.utils.data.Dataset = MultiLabelDataSet(
                self.encode_x(x), self.encode_y(y))
        elif file_name is not None:
            dataset: torch.utils.data.Dataset = IterableMultiLabelDataSet(
                file_name, self)
        else:
            raise Exception("specify either file_name or x, y")

        return TorchHelper.evaluate_model(
            self, dataset, self._get_output_layer_activation_function())

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
            dataset: torch.utils.data.Dataset = MultiLabelDataSet(x)
        elif file_name is not None:
            dataset: torch.utils.data.Dataset = IterableMultiLabelDataSet(
                file_name, self, False)
        else:
            raise Exception("specify either file_name or x")

        return TorchHelper.predict(
            self, dataset, self._get_output_layer_activation_function())

    def save_model(self, file_name: Optional[str] = None) -> None:
        TorchHelper.save_model(self, file_name)

    def write_session_info(self) -> None:
        TorchHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None) -> None:
        TorchHelper.load_model(self, file_name)

    def get_num_params(self) -> ModelSize:
        return TorchHelper.get_num_params(self)

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
