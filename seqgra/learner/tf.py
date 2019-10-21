"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from typing import List, Any
import tensorflow as tf
import os
from ast import literal_eval

from seqgra.learner.dna import DNAMultiClassClassificationLearner
from seqgra.parser.modelparser import ModelParser


class TensorFlowKerasSequentialLearner(DNAMultiClassClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)
        self.model = None

    def create_model(self) -> None:
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Flatten(input_shape=(150, 4)),
        #     tf.keras.layers.Dense(2, activation="relu"),
        #     tf.keras.layers.Dense(2, activation="softmax")
        # ])

        self.model = tf.keras.Sequential(
            [self.__get_keras_layer(operation) for operation in self.architecture.operations])

        self.model.compile(
            optimizer=self.__get_optimizer(),
            # use categorical_crossentropy for multi-class and binary_crossentropy for multi-label
            loss=self.__get_loss(),
            metrics=self.__get_metrics("training")
        )

    def __get_keras_layer(self, operation):
        if "input_shape" in operation.parameters:
            input_shape = literal_eval(
                operation.parameters["input_shape"].strip())
        else:
            input_shape = None

        name = operation.name.strip().lower()
        if name == "flatten":
            return(tf.keras.layers.Flatten(input_shape=input_shape))
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

            if "kernel_constraint" in operation.parameters:
                kernel_constraint = eval(operation.parameters["kernel_constraint"].strip(
                ))
            else:
                kernel_constraint = None

            if "bias_constraint" in operation.parameters:
                bias_constraint = eval(operation.parameters["bias_constraint"].strip(
                ))
            else:
                bias_constraint = None

            return(tf.keras.layers.Dense(units,
                                         activation=activation,
                                         use_bias=use_bias,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint,
                                         bias_constraint=bias_constraint,
                                         input_shape=input_shape))
        elif name == "conv2d":
            filters = int(operation.parameters["filters"].strip())
            kernel_size = operation.parameters["kernel_size"].strip()

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

            if "kernel_constraint" in operation.parameters:
                kernel_constraint = eval(operation.parameters["kernel_constraint"].strip(
                ))
            else:
                kernel_constraint = None

            if "bias_constraint" in operation.parameters:
                bias_constraint = eval(operation.parameters["bias_constraint"].strip(
                ))
            else:
                bias_constraint = None

            return(tf.keras.layers.Conv2D(filters,
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
                                          kernel_constraint=kernel_constraint,
                                          bias_constraint=bias_constraint,
                                          input_shape=input_shape))

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
            loss = self.optimizer_hyperparameters["loss"].lower().replace(
                "_", "").strip()
            if loss == "binarycrossentropy":
                if "from_logits" in self.optimizer_hyperparameters:
                    from_logits = bool(
                        self.optimizer_hyperparameters["from_logits"].strip())
                else:
                    from_logits = False

                if "label_smoothing" in self.optimizer_hyperparameters:
                    label_smoothing = float(
                        self.optimizer_hyperparameters["label_smoothing"].strip())
                else:
                    label_smoothing = 0.0
                return tf.keras.losses.BinaryCrossentropy(from_logits=from_logits,
                                                          label_smoothing=label_smoothing)
            elif loss == "categoricalcrossentropy":
                if "from_logits" in self.optimizer_hyperparameters:
                    from_logits = bool(
                        self.optimizer_hyperparameters["from_logits"].strip())
                else:
                    from_logits = False

                if "label_smoothing" in self.optimizer_hyperparameters:
                    label_smoothing = float(
                        self.optimizer_hyperparameters["label_smoothing"].strip())
                else:
                    label_smoothing = 0.0
                return tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits,
                                                               label_smoothing=label_smoothing)
            elif loss == "categoricalhinge":
                return tf.keras.losses.CategoricalHinge()
            elif loss == "cosinesimilarity":
                if "axis" in self.optimizer_hyperparameters:
                    axis = int(self.optimizer_hyperparameters["axis"].strip())
                else:
                    axis = -1
                return tf.keras.losses.CosineSimilarity(axis=axis)
            elif loss == "hinge":
                return tf.keras.losses.Hinge()
            elif loss == "huber":
                if "delta" in self.optimizer_hyperparameters:
                    delta = float(
                        self.optimizer_hyperparameters["delta"].strip())
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
                if "from_logits" in self.optimizer_hyperparameters:
                    from_logits = bool(
                        self.optimizer_hyperparameters["from_logits"].strip())
                else:
                    from_logits = False
                return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
            elif loss == "squaredhinge":
                return tf.keras.losses.SquaredHinge()
            else:
                raise Exception("unknown loss specified: " + loss)
        else:
            raise Exception("no loss specified")

    def __get_metrics(self, set_name):
        return [metric.name for metric in self.metrics if metric.set_name == set_name]

    def print_model_summary(self):
        self.model.summary()

    def __train_model(self):
        if self.x_train is None or self.y_train is None or self.x_val is None or self.y_val is None:
            raise Exception(
                "training and / or validation data has not been loaded")

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
                                                       mode="min", verbose=1, patience=2, min_delta=1)

        if bool(self.training_process_hyperparameters["early_stopping"]):
            callbacks = [cp_callback, tensorboard_callback, es_callback]
        else:
            callbacks = [cp_callback, tensorboard_callback]

        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=int(
                self.training_process_hyperparameters["batch_size"]),
            epochs=int(self.training_process_hyperparameters["epochs"]),
            verbose=1,
            callbacks=callbacks,
            validation_data=(self.x_val, self.y_val),
            shuffle=bool(self.training_process_hyperparameters["shuffle"])
        )

    def save_model(self, model_name: str = "final") -> None:
        self.model.save_weights(self.output_dir + model_name)

    def load_model(self, model_name: str = "final") -> None:
        latest_checkpoint = tf.train.latest_checkpoint(
            self.output_dir + model_name)

        self.__load_hyperparameters()
        self.create_model()
        self.model.load_weights(latest_checkpoint)

    def __load_hyperparameters(self):
        pass

    def evaluate_model(self, x: List[str], y: List[str]):
        val_loss, val_acc = self.model.evaluate(
            self.x_val,  self.y_val, verbose=2)
        print(val_loss)
        print(val_acc)

    def predict(self, x: Any, encode: bool = True):
        """ This is the forward calculation from x to y
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        if encode:
            x = self._encode_x(x)
        return self.model.predict(x)
