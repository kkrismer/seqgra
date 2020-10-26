"""MIT - CSAIL - Gifford Lab - seqgra

TensorFlow Keras learners

@author: Konstantin Krismer
"""
from distutils.util import strtobool
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf

from seqgra import ModelSize
from seqgra.learner import DNAMultiClassClassificationLearner
from seqgra.learner import DNAMultiLabelClassificationLearner
from seqgra.learner import ProteinMultiClassClassificationLearner
from seqgra.learner import ProteinMultiLabelClassificationLearner
from seqgra.learner.tensorflow import KerasHelper
from seqgra.model import ModelDefinition


class KerasDNAMultiClassClassificationLearner(
        DNAMultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0, silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id, silent=silent)
        KerasHelper.init_tf_memory_policy()

        gpus = tf.config.list_physical_devices("GPU")
        self.use_cuda: bool = tf.test.is_built_with_gpu_support() and \
            len(gpus) > 0 and gpu_id != -1
        if self.use_cuda:
            tf.config.set_visible_devices(gpus[gpu_id], "GPU")
            self.device_label: str = "/GPU:" + str(gpu_id)
        else:
            self.device_label: str = "/CPU:0"

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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        if x_train is not None and y_train is not None:
            training_dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x_train), self.encode_y(y_train)))
        elif file_name_train is not None:
            seq_len: int = self.get_sequence_length(file_name_train)

            def train_generator():
                return self.dataset_generator(file_name_train)
            training_dataset = tf.data.Dataset.from_generator(
                train_generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name_train or x_train, y_train")

        if x_val is not None and y_val is not None:
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x_val), self.encode_y(y_val)))
        elif file_name_val is not None:
            seq_len: int = self.get_sequence_length(file_name_val)

            def val_generator():
                return self.dataset_generator(file_name_val)
            validation_dataset = tf.data.Dataset.from_generator(
                val_generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name_val or x_val, y_val")

        KerasHelper.train_model(self, training_dataset, validation_dataset,
                                self.silent)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x), self.encode_y(y)))
        elif file_name is not None:
            seq_len: int = self.get_sequence_length(file_name)

            def generator():
                return self.dataset_generator(file_name)
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name or x, y")

        return KerasHelper.evaluate_model(self, dataset)

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
            dataset = tf.data.Dataset.from_tensor_slices((x))
        elif file_name is not None:
            seq_len: int = self.get_sequence_length(file_name)

            def generator():
                return self.dataset_generator(file_name)
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float64),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size])))
        else:
            raise Exception("specify either file_name or x")

        return KerasHelper.predict(self, dataset)

    def save_model(self, file_name: Optional[str] = None) -> None:
        KerasHelper.save_model(self, file_name)

    def write_session_info(self) -> None:
        KerasHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None) -> None:
        KerasHelper.load_model(self, file_name)

    def get_num_params(self) -> ModelSize:
        return KerasHelper.get_num_params(self)

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
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0, silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id, silent=silent)
        KerasHelper.init_tf_memory_policy()

        gpus = tf.config.list_physical_devices("GPU")
        self.use_cuda: bool = tf.test.is_built_with_gpu_support() and \
            len(gpus) > 0 and gpu_id != -1
        if self.use_cuda:
            tf.config.set_visible_devices(gpus[gpu_id], "GPU")
            self.device_label: str = "/GPU:" + str(gpu_id)
        else:
            self.device_label: str = "/CPU:0"

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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        if x_train is not None and y_train is not None:
            training_dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x_train), self.encode_y(y_train)))
        elif file_name_train is not None:
            seq_len: int = self.get_sequence_length(file_name_train)

            def train_generator():
                return self.dataset_generator(file_name_train)
            training_dataset = tf.data.Dataset.from_generator(
                train_generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name_train or x_train, y_train")

        if x_val is not None and y_val is not None:
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x_val), self.encode_y(y_val)))
        elif file_name_val is not None:
            seq_len: int = self.get_sequence_length(file_name_val)

            def val_generator():
                return self.dataset_generator(file_name_val)
            validation_dataset = tf.data.Dataset.from_generator(
                val_generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name_val or x_val, y_val")

        KerasHelper.train_model(self, training_dataset, validation_dataset,
                                self.silent)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x), self.encode_y(y)))
        elif file_name is not None:
            seq_len: int = self.get_sequence_length(file_name)

            def generator():
                return self.dataset_generator(file_name)
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name or x, y")

        return KerasHelper.evaluate_model(self, dataset)

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
            dataset = tf.data.Dataset.from_tensor_slices((x))
        elif file_name is not None:
            seq_len: int = self.get_sequence_length(file_name)

            def generator():
                return self.dataset_generator(file_name)
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float64),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size])))
        else:
            raise Exception("specify either file_name or x")

        return KerasHelper.predict(self, dataset)

    def save_model(self, file_name: Optional[str] = None) -> None:
        KerasHelper.save_model(self, file_name)

    def write_session_info(self) -> None:
        KerasHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None) -> None:
        KerasHelper.load_model(self, file_name)

    def get_num_params(self) -> ModelSize:
        return KerasHelper.get_num_params(self)

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
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0, silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id, silent=silent)
        KerasHelper.init_tf_memory_policy()

        gpus = tf.config.list_physical_devices("GPU")
        self.use_cuda: bool = tf.test.is_built_with_gpu_support() and \
            len(gpus) > 0 and gpu_id != -1
        if self.use_cuda:
            tf.config.set_visible_devices(gpus[gpu_id], "GPU")
            self.device_label: str = "/GPU:" + str(gpu_id)
        else:
            self.device_label: str = "/CPU:0"

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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        if x_train is not None and y_train is not None:
            training_dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x_train), self.encode_y(y_train)))
        elif file_name_train is not None:
            seq_len: int = self.get_sequence_length(file_name_train)

            def train_generator():
                return self.dataset_generator(file_name_train)
            training_dataset = tf.data.Dataset.from_generator(
                train_generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name_train or x_train, y_train")

        if x_val is not None and y_val is not None:
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x_val), self.encode_y(y_val)))
        elif file_name_val is not None:
            seq_len: int = self.get_sequence_length(file_name_val)

            def val_generator():
                return self.dataset_generator(file_name_val)
            validation_dataset = tf.data.Dataset.from_generator(
                val_generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name_val or x_val, y_val")

        KerasHelper.train_model(self, training_dataset, validation_dataset,
                                self.silent)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x), self.encode_y(y)))
        elif file_name is not None:
            seq_len: int = self.get_sequence_length(file_name)

            def generator():
                return self.dataset_generator(file_name)
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name or x, y")

        return KerasHelper.evaluate_model(self, dataset)

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
            dataset = tf.data.Dataset.from_tensor_slices((x))
        elif file_name is not None:
            seq_len: int = self.get_sequence_length(file_name)

            def generator():
                return self.dataset_generator(file_name)
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float64),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size])))
        else:
            raise Exception("specify either file_name or x")

        return KerasHelper.predict(self, dataset)

    def save_model(self, file_name: Optional[str] = None) -> None:
        KerasHelper.save_model(self, file_name)

    def write_session_info(self) -> None:
        KerasHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None) -> None:
        KerasHelper.load_model(self, file_name)

    def get_num_params(self) -> ModelSize:
        return KerasHelper.get_num_params(self)

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
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0, silent: bool = False) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id, silent=silent)
        KerasHelper.init_tf_memory_policy()

        gpus = tf.config.list_physical_devices("GPU")
        self.use_cuda: bool = tf.test.is_built_with_gpu_support() and \
            len(gpus) > 0 and gpu_id != -1
        if self.use_cuda:
            tf.config.set_visible_devices(gpus[gpu_id], "GPU")
            self.device_label: str = "/GPU:" + str(gpu_id)
        else:
            self.device_label: str = "/CPU:0"

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
                     file_name_train: Optional[str] = None,
                     file_name_val: Optional[str] = None,
                     x_train: Optional[List[str]] = None,
                     y_train: Optional[List[str]] = None,
                     x_val: Optional[List[str]] = None,
                     y_val: Optional[List[str]] = None) -> None:
        if x_train is not None and y_train is not None:
            training_dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x_train), self.encode_y(y_train)))
        elif file_name_train is not None:
            seq_len: int = self.get_sequence_length(file_name_train)

            def train_generator():
                return self.dataset_generator(file_name_train)
            training_dataset = tf.data.Dataset.from_generator(
                train_generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name_train or x_train, y_train")

        if x_val is not None and y_val is not None:
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x_val), self.encode_y(y_val)))
        elif file_name_val is not None:
            seq_len: int = self.get_sequence_length(file_name_val)

            def val_generator():
                return self.dataset_generator(file_name_val)
            validation_dataset = tf.data.Dataset.from_generator(
                val_generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name_val or x_val, y_val")

        KerasHelper.train_model(self, training_dataset, validation_dataset,
                                self.silent)

    def evaluate_model(self, file_name: Optional[str] = None,
                       x: Optional[List[str]] = None,
                       y: Optional[List[str]] = None):
        if x is not None and y is not None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.encode_x(x), self.encode_y(y)))
        elif file_name is not None:
            seq_len: int = self.get_sequence_length(file_name)

            def generator():
                return self.dataset_generator(file_name)
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float64, tf.bool),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size]),
                               tf.TensorShape([len(self.definition.labels)])))
        else:
            raise Exception("specify either file_name or x, y")

        return KerasHelper.evaluate_model(self, dataset)

    def predict(self, file_name: Optional[str] = None,
                x: Optional[Any] = None,
                encode: bool = True):
        if x is not None:
            if encode:
                x = self.encode_x(x)
            dataset = tf.data.Dataset.from_tensor_slices((x))
        elif file_name is not None:
            seq_len: int = self.get_sequence_length(file_name)

            def generator():
                return self.dataset_generator(file_name)
            dataset = tf.data.Dataset.from_generator(
                generator, (tf.float64),
                output_shapes=(tf.TensorShape([seq_len, self.alphabet_size])))
        else:
            raise Exception("specify either file_name or x")

        return KerasHelper.predict(self, dataset)

    def save_model(self, file_name: Optional[str] = None) -> None:
        KerasHelper.save_model(self, file_name)

    def write_session_info(self) -> None:
        KerasHelper.write_session_info(self)

    def load_model(self, file_name: Optional[str] = None) -> None:
        KerasHelper.load_model(self, file_name)

    def get_num_params(self) -> ModelSize:
        return KerasHelper.get_num_params(self)

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
