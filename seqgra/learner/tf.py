"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from typing import List, Any
import os
import sys
from ast import literal_eval
import random

import numpy as np
import tensorflow as tf
import pkg_resources

from seqgra.learner.dna import DNAMultiClassClassificationLearner
from seqgra.parser.modelparser import ModelParser


class TensorFlowKerasSequentialLearner(DNAMultiClassClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)
        self.model = None

    def create_model(self) -> None:
        self.set_seed()
        self.model = tf.keras.Sequential(
            [self.__get_keras_layer(operation) for operation in self.architecture.operations])

        self.model.compile(
            optimizer=self.__get_optimizer(),
            # use categorical_crossentropy for multi-class and binary_crossentropy for multi-label
            loss=self.__get_loss(),
            metrics=self.metrics
        )

    def print_model_summary(self):
        self.model.summary()

    def set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        # one hot encode input and labels
        encoded_x_train = self.encode_x(x_train)
        encoded_y_train = self.encode_y(y_train)
        encoded_x_val = self.encode_x(x_val)
        encoded_y_val = self.encode_y(y_val)

        if self.model is None:
            self.create_model()

        # checkpoint callback
        checkpoint_path = self.output_dir + "training/cp.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=0
        )

        # TensorBoard callback
        log_dir = self.output_dir + "logs/run"
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

        if bool(self.training_process_hyperparameters["early_stopping"]):
            callbacks = [cp_callback, tensorboard_callback, es_callback]
        else:
            callbacks = [cp_callback, tensorboard_callback]

        # training loop
        self.model.fit(
            encoded_x_train,
            encoded_y_train,
            batch_size=int(
                self.training_process_hyperparameters["batch_size"]),
            epochs=int(self.training_process_hyperparameters["epochs"]),
            verbose=1,
            callbacks=callbacks,
            validation_data=(encoded_x_val, encoded_y_val),
            shuffle=bool(self.training_process_hyperparameters["shuffle"])
        )

    def save_model(self, model_name: str = "") -> None:
        if model_name != "":
            os.makedirs(self.output_dir + model_name)
        self.model.save(self.output_dir + model_name, save_format='tf')
        self.write_session_info()

    def write_session_info(self) -> None:
        with open(self.output_dir + "session-info.txt", "w") as session_file:
            session_file.write("seqgra package version: " + pkg_resources.require("seqgra")[0].version + "\n")
            session_file.write("TensorFlow version: " + tf.__version__ + "\n")
            session_file.write("NumPy version: " + np.version.version + "\n")
            session_file.write("Python version: " + sys.version + "\n")

    def load_model(self, model_name: str = "") -> None:
        self.model = tf.keras.models.load_model(self.output_dir + model_name)

    def predict(self, x: Any, encode: bool = True):
        """ This is the forward calculation from x to y
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        if encode:
            x = self.encode_x(x)
        return self.model.predict(x)

    def get_num_params(self):
        if self.model is None:
            self.create_model()
        return 0

    def _evaluate_model(self, x: List[str], y: List[str]):
        # one hot encode input and labels
        encoded_x = self.encode_x(x)
        encoded_y = self.encode_y(y)
        return self.model.evaluate(encoded_x,  encoded_y, verbose=0)

    def __get_keras_layer(self, operation):
        if "input_shape" in operation.parameters:
            input_shape = literal_eval(
                operation.parameters["input_shape"].strip())
        else:
            input_shape = None

        name = operation.name.strip().lower()
        if name == "flatten":
            if input_shape is None:
                return(tf.keras.layers.Flatten())
            else:
                return(tf.keras.layers.Flatten(input_shape=input_shape))
        elif name == "reshape":
            target_shape = literal_eval(operation.parameters["target_shape"].strip())
            if input_shape is None:
                return(tf.keras.layers.Reshape(target_shape = target_shape))
            else:
                return(tf.keras.layers.Reshape(target_shape = target_shape,
                                               input_shape=input_shape))
        elif name == "dense":
            units = int(operation.parameters["units"].strip())

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = None

            if "use_bias" in operation.parameters:
                use_bias = bool(operation.parameters["use_bias"].strip())
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = eval(operation.parameters["kernel_initializer"].strip(
                ))
            else:
                kernel_initializer = "glorot_uniform"

            if "bias_initializer" in operation.parameters:
                bias_initializer = eval(operation.parameters["bias_initializer"].strip(
                ))
            else:
                bias_initializer = "zeros"

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = eval(operation.parameters["kernel_regularizer"].strip(
                ))
            else:
                kernel_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = eval(operation.parameters["bias_regularizer"].strip(
                ))
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = eval(operation.parameters["activity_regularizer"].strip(
                ))
            else:
                activity_regularizer = None

            if input_shape is None:
                return(tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer
                ))
            else:
                return(tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    input_shape=input_shape
                ))
        elif name == "lstm":
            units = int(operation.parameters["units"].strip())

            if "activation" in operation.parameters:
                activation = operation.parameters["activation"].strip()
            else:
                activation = "tanh"

            if "recurrent_activation" in operation.parameters:
                recurrent_activation = operation.parameters["recurrent_activation"].strip()
            else:
                recurrent_activation = "sigmoid"

            if "use_bias" in operation.parameters:
                use_bias = bool(operation.parameters["use_bias"].strip())
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = eval(operation.parameters["kernel_initializer"].strip(
                ))
            else:
                kernel_initializer = "glorot_uniform"

            if "recurrent_initializer" in operation.parameters:
                recurrent_initializer = eval(operation.parameters["recurrent_initializer"].strip(
                ))
            else:
                recurrent_initializer = "orthogonal"

            if "bias_initializer" in operation.parameters:
                bias_initializer = eval(operation.parameters["bias_initializer"].strip(
                ))
            else:
                bias_initializer = "zeros"

            if "unit_forget_bias" in operation.parameters:
                unit_forget_bias = bool(operation.parameters["unit_forget_bias"].strip())
            else:
                unit_forget_bias = True

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = eval(operation.parameters["kernel_regularizer"].strip(
                ))
            else:
                kernel_regularizer = None

            if "recurrent_regularizer" in operation.parameters:
                recurrent_regularizer = eval(operation.parameters["recurrent_regularizer"].strip(
                ))
            else:
                recurrent_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = eval(operation.parameters["bias_regularizer"].strip(
                ))
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = eval(operation.parameters["activity_regularizer"].strip(
                ))
            else:
                activity_regularizer = None

            if "dropout" in operation.parameters:
                dropout = float(operation.parameters["dropout"].strip())
            else:
                dropout = 0.0

            if "recurrent_dropout" in operation.parameters:
                recurrent_dropout = float(operation.parameters["recurrent_dropout"].strip())
            else:
                recurrent_dropout = 0.0

            if "implementation" in operation.parameters:
                implementation = int(operation.parameters["implementation"].strip())
            else:
                implementation = 2

            if "return_sequences" in operation.parameters:
                return_sequences = bool(operation.parameters["return_sequences"].strip())
            else:
                return_sequences = False

            if "return_state" in operation.parameters:
                return_state = bool(operation.parameters["return_state"].strip())
            else:
                return_state = False

            if "go_backwards" in operation.parameters:
                go_backwards = bool(operation.parameters["go_backwards"].strip())
            else:
                go_backwards = False

            if "stateful" in operation.parameters:
                stateful = bool(operation.parameters["stateful"].strip())
            else:
                stateful = False

            if "time_major" in operation.parameters:
                time_major = bool(operation.parameters["time_major"].strip())
            else:
                time_major = False

            if "unroll" in operation.parameters:
                unroll = bool(operation.parameters["unroll"].strip())
            else:
                unroll = False

            if input_shape is None:
                return(tf.keras.layers.LSTM(
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
                    unroll=unroll
                ))
            else:
                return(tf.keras.layers.LSTM(
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
                    input_shape=input_shape
                ))
        elif name == "conv1d":
            filters = int(operation.parameters["filters"].strip())
            kernel_size = literal_eval(operation.parameters["kernel_size"].strip())

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
                use_bias = bool(operation.parameters["use_bias"].strip())
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = eval(operation.parameters["kernel_initializer"].strip(
                ))
            else:
                kernel_initializer = "glorot_uniform"

            if "bias_initializer" in operation.parameters:
                bias_initializer = eval(operation.parameters["bias_initializer"].strip(
                ))
            else:
                bias_initializer = "zeros"

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = eval(operation.parameters["kernel_regularizer"].strip(
                ))
            else:
                kernel_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = eval(operation.parameters["bias_regularizer"].strip(
                ))
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = eval(operation.parameters["activity_regularizer"].strip(
                ))
            else:
                activity_regularizer = None

            if input_shape is None:
                return(tf.keras.layers.Conv1D(
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
                    activity_regularizer=activity_regularizer
                ))
            else:
                return(tf.keras.layers.Conv1D(
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
                    input_shape=input_shape
                ))
        elif name == "conv2d":
            filters = int(operation.parameters["filters"].strip())
            kernel_size = literal_eval(operation.parameters["kernel_size"].strip())

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
                use_bias = bool(operation.parameters["use_bias"].strip())
            else:
                use_bias = True

            if "kernel_initializer" in operation.parameters:
                kernel_initializer = eval(operation.parameters["kernel_initializer"].strip(
                ))
            else:
                kernel_initializer = "glorot_uniform"

            if "bias_initializer" in operation.parameters:
                bias_initializer = eval(operation.parameters["bias_initializer"].strip(
                ))
            else:
                bias_initializer = "zeros"

            if "kernel_regularizer" in operation.parameters:
                kernel_regularizer = eval(operation.parameters["kernel_regularizer"].strip(
                ))
            else:
                kernel_regularizer = None

            if "bias_regularizer" in operation.parameters:
                bias_regularizer = eval(operation.parameters["bias_regularizer"].strip(
                ))
            else:
                bias_regularizer = None

            if "activity_regularizer" in operation.parameters:
                activity_regularizer = eval(operation.parameters["activity_regularizer"].strip(
                ))
            else:
                activity_regularizer = None

            if input_shape is None:
                return(tf.keras.layers.Conv2D(
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
                    activity_regularizer=activity_regularizer
                ))
            else:
                return(tf.keras.layers.Conv2D(
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
                    input_shape=input_shape
                ))
        elif name == "globalmaxpool1d":
            return(tf.keras.layers.GlobalMaxPool1D())

    def __get_optimizer(self):
        if "optimizer" in self.optimizer_hyperparameters:
            optimizer = self.optimizer_hyperparameters["optimizer"].lower(
            ).strip()
            if optimizer == "adadelta":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(
                        self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "rho" in self.optimizer_hyperparameters:
                    rho = float(self.optimizer_hyperparameters["rho"].strip())
                else:
                    rho = 0.95

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(
                        self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07
                return tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
                                                    rho=rho,
                                                    epsilon=epsilon)
            elif optimizer == "adagrad":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(
                        self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "initial_accumulator_value" in self.optimizer_hyperparameters:
                    initial_accumulator_value = float(
                        self.optimizer_hyperparameters["initial_accumulator_value"].strip())
                else:
                    initial_accumulator_value = 0.1

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(
                        self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07
                return tf.keras.optimizers.Adagrad(learning_rate=learning_rate,
                                                   initial_accumulator_value=initial_accumulator_value,
                                                   epsilon=epsilon)
            elif optimizer == "adam":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(
                        self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "beta_1" in self.optimizer_hyperparameters:
                    beta_1 = float(
                        self.optimizer_hyperparameters["beta_1"].strip())
                else:
                    beta_1 = 0.9

                if "beta_2" in self.optimizer_hyperparameters:
                    beta_2 = float(
                        self.optimizer_hyperparameters["beta_2"].strip())
                else:
                    beta_2 = 0.999

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(
                        self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07

                if "amsgrad" in self.optimizer_hyperparameters:
                    amsgrad = bool(
                        self.optimizer_hyperparameters["amsgrad"].strip())
                else:
                    amsgrad = False
                return tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                beta_1=beta_1,
                                                beta_2=beta_2,
                                                epsilon=epsilon,
                                                amsgrad=amsgrad)
            elif optimizer == "adamax":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(
                        self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "beta_1" in self.optimizer_hyperparameters:
                    beta_1 = float(
                        self.optimizer_hyperparameters["beta_1"].strip())
                else:
                    beta_1 = 0.9

                if "beta_2" in self.optimizer_hyperparameters:
                    beta_2 = float(
                        self.optimizer_hyperparameters["beta_2"].strip())
                else:
                    beta_2 = 0.999

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(
                        self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07
                return tf.keras.optimizers.Adamax(learning_rate=learning_rate,
                                                  beta_1=beta_1,
                                                  beta_2=beta_2,
                                                  epsilon=epsilon)
            elif optimizer == "ftrl":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(
                        self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "learning_rate_power" in self.optimizer_hyperparameters:
                    learning_rate_power = float(
                        self.optimizer_hyperparameters["learning_rate_power"].strip())
                else:
                    learning_rate_power = -0.5

                if "initial_accumulator_value" in self.optimizer_hyperparameters:
                    initial_accumulator_value = float(
                        self.optimizer_hyperparameters["initial_accumulator_value"].strip())
                else:
                    initial_accumulator_value = 0.1

                if "l1_regularization_strength" in self.optimizer_hyperparameters:
                    l1_regularization_strength = float(
                        self.optimizer_hyperparameters["l1_regularization_strength"].strip())
                else:
                    l1_regularization_strength = 0.0

                if "l2_regularization_strength" in self.optimizer_hyperparameters:
                    l2_regularization_strength = float(
                        self.optimizer_hyperparameters["l2_regularization_strength"].strip())
                else:
                    l2_regularization_strength = 0.0

                if "l2_shrinkage_regularization_strength" in self.optimizer_hyperparameters:
                    l2_shrinkage_regularization_strength = float(
                        self.optimizer_hyperparameters["l2_shrinkage_regularization_strength"].strip())
                else:
                    l2_shrinkage_regularization_strength = 0.0
                return tf.keras.optimizers.Ftrl(learning_rate=learning_rate,
                                                learning_rate_power=learning_rate_power,
                                                initial_accumulator_value=initial_accumulator_value,
                                                l1_regularization_strength=l1_regularization_strength,
                                                l2_regularization_strength=l2_regularization_strength,
                                                l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength)
            elif optimizer == "nadam":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(
                        self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "beta_1" in self.optimizer_hyperparameters:
                    beta_1 = float(
                        self.optimizer_hyperparameters["beta_1"].strip())
                else:
                    beta_1 = 0.9

                if "beta_2" in self.optimizer_hyperparameters:
                    beta_2 = float(
                        self.optimizer_hyperparameters["beta_2"].strip())
                else:
                    beta_2 = 0.999

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(
                        self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07
                return tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                                 beta_1=beta_1,
                                                 beta_2=beta_2,
                                                 epsilon=epsilon)
            elif optimizer == "rmsprop":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(
                        self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "rho" in self.optimizer_hyperparameters:
                    rho = float(self.optimizer_hyperparameters["rho"].strip())
                else:
                    rho = 0.9

                if "momentum" in self.optimizer_hyperparameters:
                    momentum = float(
                        self.optimizer_hyperparameters["momentum"].strip())
                else:
                    momentum = 0.0

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(
                        self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07

                if "centered" in self.optimizer_hyperparameters:
                    centered = bool(
                        self.optimizer_hyperparameters["centered"].strip())
                else:
                    centered = False
                return tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                                   momentum=momentum,
                                                   epsilon=epsilon,
                                                   centered=centered)
            elif optimizer == "sgd":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(
                        self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "momentum" in self.optimizer_hyperparameters:
                    momentum = float(
                        self.optimizer_hyperparameters["momentum"].strip())
                else:
                    momentum = 0.0

                if "nesterov" in self.optimizer_hyperparameters:
                    nesterov = bool(
                        self.optimizer_hyperparameters["nesterov"].strip())
                else:
                    nesterov = False
                return tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                               momentum=momentum,
                                               nesterov=nesterov)
            else:
                raise Exception("unknown optimizer specified: " + optimizer)
        else:
            raise Exception("no optimizer specified")

    def __get_loss(self):
        if "loss" in self.loss_hyperparameters:
            loss = self.loss_hyperparameters["loss"].lower().replace(
                "_", "").strip()
            if loss == "binarycrossentropy":
                if "from_logits" in self.loss_hyperparameters:
                    from_logits = bool(
                        self.loss_hyperparameters["from_logits"].strip())
                else:
                    from_logits = False

                if "label_smoothing" in self.loss_hyperparameters:
                    label_smoothing = float(
                        self.loss_hyperparameters["label_smoothing"].strip())
                else:
                    label_smoothing = 0.0
                return tf.keras.losses.BinaryCrossentropy(from_logits=from_logits,
                                                          label_smoothing=label_smoothing)
            elif loss == "categoricalcrossentropy":
                if "from_logits" in self.loss_hyperparameters:
                    from_logits = bool(
                        self.loss_hyperparameters["from_logits"].strip())
                else:
                    from_logits = False

                if "label_smoothing" in self.loss_hyperparameters:
                    label_smoothing = float(
                        self.loss_hyperparameters["label_smoothing"].strip())
                else:
                    label_smoothing = 0.0
                return tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits,
                                                               label_smoothing=label_smoothing)
            elif loss == "categoricalhinge":
                return tf.keras.losses.CategoricalHinge()
            elif loss == "cosinesimilarity":
                if "axis" in self.loss_hyperparameters:
                    axis = int(self.loss_hyperparameters["axis"].strip())
                else:
                    axis = -1
                return tf.keras.losses.CosineSimilarity(axis=axis)
            elif loss == "hinge":
                return tf.keras.losses.Hinge()
            elif loss == "huber":
                if "delta" in self.loss_hyperparameters:
                    delta = float(
                        self.loss_hyperparameters["delta"].strip())
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
                if "from_logits" in self.loss_hyperparameters:
                    from_logits = bool(
                        self.loss_hyperparameters["from_logits"].strip())
                else:
                    from_logits = False
                return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
            elif loss == "squaredhinge":
                return tf.keras.losses.SquaredHinge()
            else:
                raise Exception("unknown loss specified: " + loss)
        else:
            raise Exception("no loss specified")
