"""MIT - CSAIL - Gifford Lab - seqgra

TensorFlow Keras learner helper class

@author: Konstantin Krismer
"""
from ast import literal_eval
from typing import List, Any
import os
import sys
import random
import pkg_resources

import tensorflow as tf
import numpy as np

from seqgra.learner import Learner
from seqgra.model.model import Architecture


class KerasHelper:
    @staticmethod
    def to_bool(x: str) -> bool:
        x = x.strip()
        if x == "True":
            return True
        elif x == "False":
            return False
        else:
            raise Exception("'" + str(x) +
                            "' must be either 'True' or 'False'")

    @staticmethod
    def create_model(learner: Learner) -> None:
        arch: Architecture = learner.definition.architecture
        path = arch.external_model_path
        learner.set_seed()

        if arch.operations is not None:
            learner.model = tf.keras.Sequential(
                [KerasHelper.get_keras_layer(operation)
                 for operation in arch.operations])

            for i in range(len(arch.operations)):
                custom_weights = KerasHelper.load_custom_weights(
                    arch.operations[i])
                if custom_weights is not None:
                    learner.model.layers[i].set_weights(custom_weights)
        elif path is not None:
            if arch.external_model_format == "keras-h5-whole-model":
                if os.path.isfile(path):
                    learner.model = tf.keras.models.load_model(path)
                else:
                    raise Exception(".h5 file does not exist: " + path)
            elif arch.external_model_format == "keras-tf-whole-model":
                if os.path.isdir(path):
                    learner.model = tf.keras.models.load_model(path)
                else:
                    raise Exception("TF saved model directory does not exist: " +
                                    path)
            elif arch.external_model_format == "keras-json-architecture-only":
                if os.path.isfile(path):
                    with open(path, "r") as json_config_file:
                        json_config = json_config_file.read()
                        learner.model = tf.keras.models.model_from_json(
                            json_config)
                else:
                    raise Exception(".json file does not exist: " + path)
            elif arch.external_model_format == "keras-yaml-architecture-only":
                if os.path.isfile(path):
                    with open(path, "r") as yaml_config_file:
                        yaml_config = yaml_config_file.read()
                        learner.model = tf.keras.models.model_from_yaml(
                            yaml_config)
                else:
                    raise Exception(".yaml file does not exist: " + path)
        else:
            raise Exception("neither internal nor external architecture "
                            "definition provided")

        if arch.external_model_format is None or \
           arch.external_model_format == "keras-yaml-architecture-only" or \
           arch.external_model_format == "keras-json-architecture-only":
            if learner.definition.optimizer_hyperparameters is not None and \
               learner.definition.loss_hyperparameters is not None and \
               learner.metrics is not None:
                local_metrics = learner.metrics[learner.metrics != "loss"]
                if type(local_metrics) != "list":
                    local_metrics = [local_metrics]
                learner.model.compile(
                    optimizer=KerasHelper.get_optimizer(
                        learner.definition.optimizer_hyperparameters),
                    # use categorical_crossentropy for multi-class and
                    # binary_crossentropy for multi-label
                    loss=KerasHelper.get_loss(
                        learner.definition.loss_hyperparameters),
                    metrics=local_metrics
                )
            else:
                raise Exception("optimizer, loss or metrics undefined")

    @staticmethod
    def print_model_summary(learner: Learner):
        learner.model.summary()

    @staticmethod
    def set_seed(learner: Learner) -> None:
        random.seed(learner.definition.seed)
        np.random.seed(learner.definition.seed)
        tf.random.set_seed(learner.definition.seed)

    @staticmethod
    def train_model(learner: Learner,
                    x_train: List[str], y_train: List[str],
                    x_val: List[str], y_val: List[str]) -> None:
        # one hot encode input and labels
        encoded_x_train = learner.encode_x(x_train)
        encoded_y_train = learner.encode_y(y_train)
        encoded_x_val = learner.encode_x(x_val)
        encoded_y_val = learner.encode_y(y_val)

        if learner.model is None:
            learner.create_model()

        # checkpoint callback
        checkpoint_path = learner.output_dir + "training/cp.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=0
        )

        # TensorBoard callback
        log_dir = learner.output_dir + "logs/run"
        os.makedirs(log_dir)
        log_dir = log_dir.replace("/", "\\")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )

        # early stopping callback
        es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                       mode="min",
                                                       verbose=1,
                                                       patience=2,
                                                       min_delta=0)

        if bool(learner.definition.training_process_hyperparameters["early_stopping"]):
            callbacks = [cp_callback, tensorboard_callback, es_callback]
        else:
            callbacks = [cp_callback, tensorboard_callback]

        # training loop
        learner.model.fit(
            encoded_x_train,
            encoded_y_train,
            batch_size=int(
                learner.definition.training_process_hyperparameters["batch_size"]),
            epochs=int(
                learner.definition.training_process_hyperparameters["epochs"]),
            verbose=1,
            callbacks=callbacks,
            validation_data=(encoded_x_val, encoded_y_val),
            shuffle=bool(
                learner.definition.training_process_hyperparameters["shuffle"])
        )

    @staticmethod
    def save_model(learner: Learner, model_name: str = "") -> None:
        if model_name != "":
            os.makedirs(learner.output_dir + model_name)

        # save whole model (TensorFlow format)
        learner.model.save(learner.output_dir + model_name, save_format="tf")

        # save whole model (HDF5)
        learner.model.save(learner.output_dir + model_name + "/saved_model.h5",
                           save_format="h5")

        # save architecture only (YAML)
        yaml_model = learner.model.to_yaml()
        with open(learner.output_dir + model_name + "/model-architecture.yaml", "w") as yaml_file:
            yaml_file.write(yaml_model)

        # save architecture only (JSON)
        json_model = learner.model.to_json()
        with open(learner.output_dir + model_name + "/model-architecture.json", "w") as json_file:
            json_file.write(json_model)

        # save session info
        learner.write_session_info()

    @staticmethod
    def write_session_info(learner: Learner) -> None:
        with open(learner.output_dir + "session-info.txt", "w") as session_file:
            session_file.write("seqgra package version: " +
                               pkg_resources.require("seqgra")[0].version + "\n")
            session_file.write("TensorFlow version: " + tf.__version__ + "\n")
            session_file.write("NumPy version: " + np.version.version + "\n")
            session_file.write("Python version: " + sys.version + "\n")

    @staticmethod
    def load_model(learner: Learner, model_name: str = "") -> None:
        learner.model = tf.keras.models.load_model(
            learner.output_dir + model_name)

    @staticmethod
    def predict(learner: Learner, x: Any, encode: bool = True):
        """ This is the forward calculation from x to y
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        if encode:
            x = learner.encode_x(x)
        return learner.model.predict(x)

    @staticmethod
    def get_num_params(learner: Learner):
        if learner.model is None:
            learner.create_model()
        return 0

    @staticmethod
    def evaluate_model(learner: Learner, x: List[str], y: List[str]):
        # one hot encode input and labels
        encoded_x = learner.encode_x(x)
        encoded_y = learner.encode_y(y)
        loss, accuracy = learner.model.evaluate(encoded_x, encoded_y,
                                                verbose=0)
        return {"loss": loss, "accuracy": accuracy}

    @staticmethod
    def get_keras_layer(operation):
        if "input_shape" in operation.parameters:
            input_shape = literal_eval(
                operation.parameters["input_shape"].strip())
        else:
            input_shape = None

        if "trainable" in operation.parameters:
            trainable = KerasHelper.to_bool(
                operation.parameters["trainable"])
        else:
            trainable = True

        name = operation.name.strip().lower()
        if name == "flatten":
            if input_shape is None:
                return tf.keras.layers.Flatten()
            else:
                return tf.keras.layers.Flatten(input_shape=input_shape)
        elif name == "reshape":
            target_shape = literal_eval(
                operation.parameters["target_shape"].strip())
            if input_shape is None:
                return tf.keras.layers.Reshape(target_shape=target_shape)
            else:
                return tf.keras.layers.Reshape(target_shape=target_shape,
                                               input_shape=input_shape)
        elif name == "dense":
            units = int(operation.parameters["units"].strip())

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = None

            if "use_bias" in operation.parameters:
                use_bias = KerasHelper.to_bool(
                    operation.parameters["use_bias"])
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = \
                    eval(operation.parameters["kernel_initializer"].strip())
            else:
                kernel_initializer = "glorot_uniform"

            if "bias_initializer" in operation.parameters:
                bias_initializer = \
                    eval(operation.parameters["bias_initializer"].strip())
            else:
                bias_initializer = "zeros"

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = \
                    eval(operation.parameters["kernel_regularizer"].strip())
            else:
                kernel_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = \
                    eval(operation.parameters["bias_regularizer"].strip())
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = \
                    eval(operation.parameters["activity_regularizer"].strip())
            else:
                activity_regularizer = None

            if input_shape is None:
                return tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    trainable=trainable
                )
            else:
                return tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    trainable=trainable,
                    input_shape=input_shape
                )
        elif name == "lstm":
            units = int(operation.parameters["units"].strip())

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = "tanh"

            if "recurrent_activation" in operation.parameters:
                recurrent_activation = \
                    operation.parameters["recurrent_activation"].strip()
            else:
                recurrent_activation = "sigmoid"

            if "use_bias" in operation.parameters:
                use_bias = KerasHelper.to_bool(
                    operation.parameters["use_bias"])
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = \
                    eval(operation.parameters["kernel_initializer"].strip())
            else:
                kernel_initializer = "glorot_uniform"

            if "recurrent_initializer" in operation.parameters:
                recurrent_initializer = \
                    eval(operation.parameters["recurrent_initializer"].strip())
            else:
                recurrent_initializer = "orthogonal"

            if "bias_initializer" in operation.parameters:
                bias_initializer = \
                    eval(operation.parameters["bias_initializer"].strip())
            else:
                bias_initializer = "zeros"

            if "unit_forget_bias" in operation.parameters:
                unit_forget_bias = \
                    KerasHelper.to_bool(
                        operation.parameters["unit_forget_bias"])
            else:
                unit_forget_bias = True

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = \
                    eval(operation.parameters["kernel_regularizer"].strip())
            else:
                kernel_regularizer = None

            if "recurrent_regularizer" in operation.parameters:
                recurrent_regularizer = \
                    eval(operation.parameters["recurrent_regularizer"].strip())
            else:
                recurrent_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = \
                    eval(operation.parameters["bias_regularizer"].strip())
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = \
                    eval(operation.parameters["activity_regularizer"].strip())
            else:
                activity_regularizer = None

            if "dropout" in operation.parameters:
                dropout = float(operation.parameters["dropout"].strip())
            else:
                dropout = 0.0

            if "recurrent_dropout" in operation.parameters:
                recurrent_dropout = \
                    float(operation.parameters["recurrent_dropout"].strip())
            else:
                recurrent_dropout = 0.0

            if "implementation" in operation.parameters:
                implementation = \
                    int(operation.parameters["implementation"].strip())
            else:
                implementation = 2

            if "return_sequences" in operation.parameters:
                return_sequences = \
                    KerasHelper.to_bool(
                        operation.parameters["return_sequences"])
            else:
                return_sequences = False

            if "return_state" in operation.parameters:
                return_state = \
                    KerasHelper.to_bool(operation.parameters["return_state"])
            else:
                return_state = False

            if "go_backwards" in operation.parameters:
                go_backwards = \
                    KerasHelper.to_bool(operation.parameters["go_backwards"])
            else:
                go_backwards = False

            if "stateful" in operation.parameters:
                stateful = KerasHelper.to_bool(
                    operation.parameters["stateful"])
            else:
                stateful = False

            if "time_major" in operation.parameters:
                time_major = KerasHelper.to_bool(
                    operation.parameters["time_major"])
            else:
                time_major = False

            if "unroll" in operation.parameters:
                unroll = KerasHelper.to_bool(operation.parameters["unroll"])
            else:
                unroll = False

            if "bidirectional" in operation.parameters:
                bidirectional = KerasHelper.to_bool(
                    operation.parameters["bidirectional"])
            else:
                bidirectional = False

            if input_shape is None:
                lstm_layer = tf.keras.layers.LSTM(
                    units,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    unit_forget_bias=unit_forget_bias,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    implementation=implementation,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    go_backwards=go_backwards,
                    stateful=stateful,
                    time_major=time_major,
                    unroll=unroll,
                    trainable=trainable
                )
            else:
                lstm_layer = tf.keras.layers.LSTM(
                    units,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    unit_forget_bias=unit_forget_bias,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    implementation=implementation,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    go_backwards=go_backwards,
                    stateful=stateful,
                    time_major=time_major,
                    unroll=unroll,
                    trainable=trainable,
                    input_shape=input_shape
                )
            if bidirectional:
                if "merge_mode" in operation.parameters:
                    merge_mode = operation.parameters["merge_mode"].strip()
                else:
                    merge_mode = "concat"

                return tf.keras.layers.Bidirectional(lstm_layer,
                                                     merge_mode=merge_mode)
            else:
                return lstm_layer
        elif name == "conv1d":
            filters = int(operation.parameters["filters"].strip())
            kernel_size = \
                literal_eval(operation.parameters["kernel_size"].strip())

            if "strides" in operation.parameters:
                strides = literal_eval(operation.parameters["strides"].strip())
            else:
                strides = 1

            if "padding" in operation.parameters:
                padding = operation.parameters["padding"].strip()
            else:
                padding = "valid"

            if "data_format" in operation.parameters:
                data_format = operation.parameters["data_format"].strip()
            else:
                data_format = "channels_last"

            if "dilation_rate" in operation.parameters:
                dilation_rate = operation.parameters["dilation_rate"].strip()
            else:
                dilation_rate = 1

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = None

            if "use_bias" in operation.parameters:
                use_bias = KerasHelper.to_bool(
                    operation.parameters["use_bias"])
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = \
                    eval(operation.parameters["kernel_initializer"].strip())
            else:
                kernel_initializer = "glorot_uniform"

            if "bias_initializer" in operation.parameters:
                bias_initializer = \
                    eval(operation.parameters["bias_initializer"].strip())
            else:
                bias_initializer = "zeros"

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = \
                    eval(operation.parameters["kernel_regularizer"].strip())
            else:
                kernel_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = \
                    eval(operation.parameters["bias_regularizer"].strip())
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = \
                    eval(operation.parameters["activity_regularizer"].strip())
            else:
                activity_regularizer = None

            if input_shape is None:
                return tf.keras.layers.Conv1D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    trainable=trainable
                )
            else:
                return tf.keras.layers.Conv1D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    trainable=trainable,
                    input_shape=input_shape
                )
        elif name == "conv2d":
            filters = int(operation.parameters["filters"].strip())
            kernel_size = \
                literal_eval(operation.parameters["kernel_size"].strip())

            if "strides" in operation.parameters:
                strides = literal_eval(operation.parameters["strides"].strip())
            else:
                strides = (1, 1)

            if "padding" in operation.parameters:
                padding = operation.parameters["padding"].strip()
            else:
                padding = "valid"

            if "data_format" in operation.parameters:
                data_format = operation.parameters["data_format"].strip()
            else:
                data_format = None

            if "dilation_rate" in operation.parameters:
                dilation_rate = operation.parameters["dilation_rate"].strip()
            else:
                dilation_rate = (1, 1)

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = None

            if "use_bias" in operation.parameters:
                use_bias = KerasHelper.to_bool(
                    operation.parameters["use_bias"])
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = \
                    eval(operation.parameters["kernel_initializer"].strip())
            else:
                kernel_initializer = "glorot_uniform"

            if "bias_initializer" in operation.parameters:
                bias_initializer = \
                    eval(operation.parameters["bias_initializer"].strip())
            else:
                bias_initializer = "zeros"

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = \
                    eval(operation.parameters["kernel_regularizer"].strip())
            else:
                kernel_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = \
                    eval(operation.parameters["bias_regularizer"].strip())
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = \
                    eval(operation.parameters["activity_regularizer"].strip())
            else:
                activity_regularizer = None

            if input_shape is None:
                return tf.keras.layers.Conv2D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    trainable=trainable
                )
            else:
                return tf.keras.layers.Conv2D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    trainable=trainable,
                    input_shape=input_shape
                )
        elif name == "globalmaxpool1d":
            return tf.keras.layers.GlobalMaxPool1D()
        elif name == "maxpool1d":
            if "pool_size" in operation.parameters:
                pool_size = int(operation.parameters["pool_size"].strip())
            else:
                pool_size = 2

            if "strides" in operation.parameters:
                strides = int(operation.parameters["strides"].strip())
            else:
                strides = None

            if "padding" in operation.parameters:
                padding = operation.parameters["padding"].strip()
            else:
                padding = "valid"

            return tf.keras.layers.MaxPool1D(pool_size=pool_size,
                                             strides=strides,
                                             padding=padding)
        elif name == "dropout":
            rate = float(operation.parameters["rate"].strip())

            return tf.keras.layers.Dropout(rate)

    @staticmethod
    def load_custom_weights(operation):
        if "weights_file" in operation.parameters:
            weights_file = operation.parameters["weights_file"].strip()
        else:
            return None

        # check if file exists
        if os.path.isfile(weights_file):
            return np.load(weights_file, allow_pickle=True)
        else:
            raise Exception("weights_file (" + weights_file + ") not "
                            "found")

    @staticmethod
    def get_optimizer(optimizer_hyperparameters):
        if "optimizer" in optimizer_hyperparameters:
            optimizer = \
                optimizer_hyperparameters["optimizer"].lower().strip()

            if "learning_rate" in optimizer_hyperparameters:
                learning_rate = float(
                    optimizer_hyperparameters["learning_rate"].strip())
            else:
                learning_rate = 0.001

            if "beta_1" in optimizer_hyperparameters:
                beta_1 = float(
                    optimizer_hyperparameters["beta_1"].strip())
            else:
                beta_1 = 0.9

            if "beta_2" in optimizer_hyperparameters:
                beta_2 = float(
                    optimizer_hyperparameters["beta_2"].strip())
            else:
                beta_2 = 0.999

            if "epsilon" in optimizer_hyperparameters:
                epsilon = float(
                    optimizer_hyperparameters["epsilon"].strip())
            else:
                epsilon = 1e-07

            if "clipnorm" in optimizer_hyperparameters:
                clipnorm = float(
                    optimizer_hyperparameters["clipnorm"].strip())
            else:
                clipnorm = None

            if "clipvalue" in optimizer_hyperparameters:
                clipvalue = float(
                    optimizer_hyperparameters["clipvalue"].strip())
            else:
                clipvalue = None

            if optimizer == "adadelta":
                if "rho" in optimizer_hyperparameters:
                    rho = float(optimizer_hyperparameters["rho"].strip())
                else:
                    rho = 0.95

                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Adadelta(
                        learning_rate=learning_rate,
                        rho=rho,
                        epsilon=epsilon)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Adadelta(
                        learning_rate=learning_rate,
                        rho=rho,
                        epsilon=epsilon,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Adadelta(
                        learning_rate=learning_rate,
                        rho=rho,
                        epsilon=epsilon,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Adadelta(
                        learning_rate=learning_rate,
                        rho=rho,
                        epsilon=epsilon,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "adagrad":
                if "initial_accumulator_value" in optimizer_hyperparameters:
                    initial_accumulator_value = float(
                        optimizer_hyperparameters["initial_accumulator_value"].strip())
                else:
                    initial_accumulator_value = 0.1

                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Adagrad(
                        learning_rate=learning_rate,
                        initial_accumulator_value=initial_accumulator_value,
                        epsilon=epsilon)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Adagrad(
                        learning_rate=learning_rate,
                        initial_accumulator_value=initial_accumulator_value,
                        epsilon=epsilon,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Adagrad(
                        learning_rate=learning_rate,
                        initial_accumulator_value=initial_accumulator_value,
                        epsilon=epsilon,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Adagrad(
                        learning_rate=learning_rate,
                        initial_accumulator_value=initial_accumulator_value,
                        epsilon=epsilon,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "adam":
                if "amsgrad" in optimizer_hyperparameters:
                    amsgrad = KerasHelper.to_bool(
                        optimizer_hyperparameters["amsgrad"])
                else:
                    amsgrad = False

                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        amsgrad=amsgrad)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        amsgrad=amsgrad,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        amsgrad=amsgrad,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        amsgrad=amsgrad,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "adamax":
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Adamax(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Adamax(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Adamax(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Adamax(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "ftrl":
                if "learning_rate_power" in optimizer_hyperparameters:
                    learning_rate_power = float(
                        optimizer_hyperparameters["learning_rate_power"].strip())
                else:
                    learning_rate_power = -0.5

                if "initial_accumulator_value" in optimizer_hyperparameters:
                    initial_accumulator_value = float(
                        optimizer_hyperparameters["initial_accumulator_value"].strip())
                else:
                    initial_accumulator_value = 0.1

                if "l1_regularization_strength" in optimizer_hyperparameters:
                    l1_regularization_strength = float(
                        optimizer_hyperparameters["l1_regularization_strength"].strip())
                else:
                    l1_regularization_strength = 0.0

                if "l2_regularization_strength" in optimizer_hyperparameters:
                    l2_regularization_strength = float(
                        optimizer_hyperparameters["l2_regularization_strength"].strip())
                else:
                    l2_regularization_strength = 0.0

                if "l2_shrinkage_regularization_strength" in optimizer_hyperparameters:
                    l2_shrinkage_regularization_strength = float(
                        optimizer_hyperparameters["l2_shrinkage_regularization_strength"].strip())
                else:
                    l2_shrinkage_regularization_strength = 0.0

                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Ftrl(
                        learning_rate=learning_rate,
                        learning_rate_power=learning_rate_power,
                        initial_accumulator_value=initial_accumulator_value,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Ftrl(
                        learning_rate=learning_rate,
                        learning_rate_power=learning_rate_power,
                        initial_accumulator_value=initial_accumulator_value,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Ftrl(
                        learning_rate=learning_rate,
                        learning_rate_power=learning_rate_power,
                        initial_accumulator_value=initial_accumulator_value,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Ftrl(
                        learning_rate=learning_rate,
                        learning_rate_power=learning_rate_power,
                        initial_accumulator_value=initial_accumulator_value,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "nadam":
                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.Nadam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.Nadam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.Nadam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.Nadam(
                        learning_rate=learning_rate,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        epsilon=epsilon,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "rmsprop":
                if "rho" in optimizer_hyperparameters:
                    rho = float(optimizer_hyperparameters["rho"].strip())
                else:
                    rho = 0.9

                if "momentum" in optimizer_hyperparameters:
                    momentum = float(
                        optimizer_hyperparameters["momentum"].strip())
                else:
                    momentum = 0.0

                if "centered" in optimizer_hyperparameters:
                    centered = KerasHelper.to_bool(
                        optimizer_hyperparameters["centered"])
                else:
                    centered = False

                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.RMSprop(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        epsilon=epsilon,
                        centered=centered)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.RMSprop(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        epsilon=epsilon,
                        centered=centered,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.RMSprop(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        epsilon=epsilon,
                        centered=centered,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.RMSprop(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        epsilon=epsilon,
                        centered=centered,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            elif optimizer == "sgd":
                if "momentum" in optimizer_hyperparameters:
                    momentum = float(
                        optimizer_hyperparameters["momentum"].strip())
                else:
                    momentum = 0.0

                if "nesterov" in optimizer_hyperparameters:
                    nesterov = KerasHelper.to_bool(
                        optimizer_hyperparameters["nesterov"])
                else:
                    nesterov = False

                if clipnorm is None and clipvalue is None:
                    return tf.keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                        centered=centered)
                elif clipnorm is None and clipvalue is not None:
                    return tf.keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                        clipvalue=clipvalue)
                elif clipnorm is not None and clipvalue is None:
                    return tf.keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                        clipnorm=clipnorm)
                else:
                    return tf.keras.optimizers.SGD(
                        learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov,
                        clipnorm=clipnorm,
                        clipvalue=clipvalue)
            else:
                raise Exception("unknown optimizer specified: " + optimizer)
        else:
            raise Exception("no optimizer specified")

    @staticmethod
    def get_loss(loss_hyperparameters):
        if "loss" in loss_hyperparameters:
            loss = loss_hyperparameters["loss"].lower().replace(
                "_", "").strip()
            if loss == "binarycrossentropy":
                if "from_logits" in loss_hyperparameters:
                    from_logits = KerasHelper.to_bool(
                        loss_hyperparameters["from_logits"])
                else:
                    from_logits = False

                if "label_smoothing" in loss_hyperparameters:
                    label_smoothing = float(
                        loss_hyperparameters["label_smoothing"].strip())
                else:
                    label_smoothing = 0.0
                return tf.keras.losses.BinaryCrossentropy(
                    from_logits=from_logits,
                    label_smoothing=label_smoothing)
            elif loss == "categoricalcrossentropy":
                if "from_logits" in loss_hyperparameters:
                    from_logits = KerasHelper.to_bool(
                        loss_hyperparameters["from_logits"])
                else:
                    from_logits = False

                if "label_smoothing" in loss_hyperparameters:
                    label_smoothing = float(
                        loss_hyperparameters["label_smoothing"].strip())
                else:
                    label_smoothing = 0.0
                return tf.keras.losses.CategoricalCrossentropy(
                    from_logits=from_logits,
                    label_smoothing=label_smoothing)
            elif loss == "categoricalhinge":
                return tf.keras.losses.CategoricalHinge()
            elif loss == "cosinesimilarity":
                if "axis" in loss_hyperparameters:
                    axis = int(loss_hyperparameters["axis"].strip())
                else:
                    axis = -1
                return tf.keras.losses.CosineSimilarity(axis=axis)
            elif loss == "hinge":
                return tf.keras.losses.Hinge()
            elif loss == "huber":
                if "delta" in loss_hyperparameters:
                    delta = float(
                        loss_hyperparameters["delta"].strip())
                else:
                    delta = 1.0
                return tf.keras.losses.Huber(delta=delta)
            elif loss == "kldivergence" or loss == "kld":
                return tf.keras.losses.KLDivergence()
            elif loss == "logcosh":
                return tf.keras.losses.LogCosh()
            elif loss == "meanabsoluteerror" or loss == "mae":
                return tf.keras.losses.MeanAbsoluteError()
            elif loss == "meanabsolutepercentageerror" or loss == "mape":
                return tf.keras.losses.MeanAbsolutePercentageError()
            elif loss == "meansquarederror" or loss == "mse":
                return tf.keras.losses.MeanSquaredError()
            elif loss == "meansquaredlogarithmicerror" or loss == "msle":
                return tf.keras.losses.MeanSquaredLogarithmicError()
            elif loss == "poisson":
                return tf.keras.losses.Poisson()
            elif loss == "sparsecategoricalcrossentropy":
                if "from_logits" in loss_hyperparameters:
                    from_logits = KerasHelper.to_bool(
                        loss_hyperparameters["from_logits"])
                else:
                    from_logits = False
                return tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=from_logits)
            elif loss == "squaredhinge":
                return tf.keras.losses.SquaredHinge()
            else:
                raise Exception("unknown loss specified: " + loss)
        else:
            raise Exception("no loss specified")
