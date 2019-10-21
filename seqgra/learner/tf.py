"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from typing import List, Any
import tensorflow as tf
import os

from seqgra.learner.dna import DNAMultiClassClassificationLearner
from seqgra.parser.modelparser import ModelParser

class TensorFlowKerasSequentialLearner(DNAMultiClassClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)
        self.model = None
    
    def create_model(self) -> None:
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(150, 4)),
            tf.keras.layers.Dense(2, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax")
        ])

        self.model.compile(
            optimizer = self.__get_optimizer(),
            loss = "categorical_crossentropy", # binary_crossentropy for multi-label
            metrics = ["accuracy"]
        )

    def __get_optimizer(self):
        if "optimizer" in self.optimizer_hyperparameters:
            optimizer = self.optimizer_hyperparameters["optimizer"].lower().strip()
            if optimizer == "adadelta":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "rho" in self.optimizer_hyperparameters:
                    rho = float(self.optimizer_hyperparameters["rho"].strip())
                else:
                    rho = 0.95

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07
                return tf.keras.optimizers.Adadelta(learning_rate = learning_rate,
                                                    rho = rho,
                                                    epsilon = epsilon)
            elif optimizer == "adagrad":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "initial_accumulator_value" in self.optimizer_hyperparameters:
                    initial_accumulator_value = float(self.optimizer_hyperparameters["initial_accumulator_value"].strip())
                else:
                    initial_accumulator_value = 0.1

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07
                return tf.keras.optimizers.Adagrad(learning_rate = learning_rate,
                                                   initial_accumulator_value = initial_accumulator_value,
                                                   epsilon = epsilon)
            elif optimizer == "adam":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "beta_1" in self.optimizer_hyperparameters:
                    beta_1 = float(self.optimizer_hyperparameters["beta_1"].strip())
                else:
                    beta_1 = 0.9

                if "beta_2" in self.optimizer_hyperparameters:
                    beta_2 = float(self.optimizer_hyperparameters["beta_2"].strip())
                else:
                    beta_2 = 0.999

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07

                if "amsgrad" in self.optimizer_hyperparameters:
                    amsgrad = bool(self.optimizer_hyperparameters["amsgrad"].strip())
                else:
                    amsgrad = False
                return tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                                beta_1 = beta_1,
                                                beta_2 = beta_2,
                                                epsilon = epsilon,
                                                amsgrad = amsgrad)
            elif optimizer == "adamax":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "beta_1" in self.optimizer_hyperparameters:
                    beta_1 = float(self.optimizer_hyperparameters["beta_1"].strip())
                else:
                    beta_1 = 0.9

                if "beta_2" in self.optimizer_hyperparameters:
                    beta_2 = float(self.optimizer_hyperparameters["beta_2"].strip())
                else:
                    beta_2 = 0.999

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07
                return tf.keras.optimizers.Adamax(learning_rate = learning_rate,
                                                  beta_1 = beta_1,
                                                  beta_2 = beta_2, 
                                                  epsilon = epsilon)
            elif optimizer == "ftrl":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "learning_rate_power" in self.optimizer_hyperparameters:
                    learning_rate_power = float(self.optimizer_hyperparameters["learning_rate_power"].strip())
                else:
                    learning_rate_power = -0.5

                if "initial_accumulator_value" in self.optimizer_hyperparameters:
                    initial_accumulator_value = float(self.optimizer_hyperparameters["initial_accumulator_value"].strip())
                else:
                    initial_accumulator_value = 0.1

                if "l1_regularization_strength" in self.optimizer_hyperparameters:
                    l1_regularization_strength = float(self.optimizer_hyperparameters["l1_regularization_strength"].strip())
                else:
                    l1_regularization_strength = 0.0

                if "l2_regularization_strength" in self.optimizer_hyperparameters:
                    l2_regularization_strength = float(self.optimizer_hyperparameters["l2_regularization_strength"].strip())
                else:
                    l2_regularization_strength = 0.0

                if "l2_shrinkage_regularization_strength" in self.optimizer_hyperparameters:
                    l2_shrinkage_regularization_strength = float(self.optimizer_hyperparameters["l2_shrinkage_regularization_strength"].strip())
                else:
                    l2_shrinkage_regularization_strength = 0.0
                return tf.keras.optimizers.Ftrl(learning_rate = learning_rate,
                                                learning_rate_power = learning_rate_power, 
                                                initial_accumulator_value = initial_accumulator_value,
                                                l1_regularization_strength = l1_regularization_strength,
                                                l2_regularization_strength = l2_regularization_strength,
                                                l2_shrinkage_regularization_strength = l2_shrinkage_regularization_strength)
            elif optimizer == "nadam":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "beta_1" in self.optimizer_hyperparameters:
                    beta_1 = float(self.optimizer_hyperparameters["beta_1"].strip())
                else:
                    beta_1 = 0.9

                if "beta_2" in self.optimizer_hyperparameters:
                    beta_2 = float(self.optimizer_hyperparameters["beta_2"].strip())
                else:
                    beta_2 = 0.999

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07
                return tf.keras.optimizers.Nadam(learning_rate = learning_rate,
                                                 beta_1 = beta_1,
                                                 beta_2 = beta_2,
                                                 epsilon = epsilon)
            elif optimizer == "rmsprop":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "rho" in self.optimizer_hyperparameters:
                    rho = float(self.optimizer_hyperparameters["rho"].strip())
                else:
                    rho = 0.9

                if "momentum" in self.optimizer_hyperparameters:
                    momentum = float(self.optimizer_hyperparameters["momentum"].strip())
                else:
                    momentum = 0.0

                if "epsilon" in self.optimizer_hyperparameters:
                    epsilon = float(self.optimizer_hyperparameters["epsilon"].strip())
                else:
                    epsilon = 1e-07
                    
                if "centered" in self.optimizer_hyperparameters:
                    centered = bool(self.optimizer_hyperparameters["centered"].strip())
                else:
                    centered = False
                return tf.keras.optimizers.RMSprop(learning_rate = learning_rate,
                                                   momentum = momentum,
                                                   epsilon = epsilon, 
                                                   centered = centered)
            elif optimizer == "SGD":
                if "learning_rate" in self.optimizer_hyperparameters:
                    learning_rate = float(self.optimizer_hyperparameters["learning_rate"].strip())
                else:
                    learning_rate = 0.001

                if "momentum" in self.optimizer_hyperparameters:
                    momentum = float(self.optimizer_hyperparameters["momentum"].strip())
                else:
                    momentum = 0.0
                    
                if "nesterov" in self.optimizer_hyperparameters:
                    nesterov = bool(self.optimizer_hyperparameters["nesterov"].strip())
                else:
                    nesterov = False
                return tf.keras.optimizers.SGD(learning_rate = learning_rate,
                                               momentum = momentum, 
                                               nesterov = nesterov)
            else:
                raise Exception("unknown optimizer specified: " + optimizer)
        else:
            raise Exception("no optimizer specified")

    def print_model_summary(self):
        self.model.summary()

    def __train_model(self):
        if self.x_train is None or self.y_train is None or self.x_val is None or self.y_val is None:
            raise Exception("training and / or validation data has not been loaded")

        if self.model is None:
            self.create_model()

        # checkpoint callback
        checkpoint_path = self.output_dir + "training/cp.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = checkpoint_path,
            save_weights_only = True,
            verbose = 0
        )
        
        # TensorBoard callback
        log_dir = self.output_dir + "logs/run"
        os.makedirs(log_dir)
        log_dir = log_dir.replace("/", "\\")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir = log_dir,
            histogram_freq = 0,
            write_graph = True,
            write_images = True
        )

        # early stopping callback
        es_callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
            mode = "min", verbose = 1, patience = 2, min_delta = 1)

        if bool(self.training_process_hyperparameters["early_stopping"]):
            callbacks = [cp_callback, tensorboard_callback, es_callback]
        else:
            callbacks = [cp_callback, tensorboard_callback]

        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size = int(self.training_process_hyperparameters["batch_size"]),
            epochs = int(self.training_process_hyperparameters["epochs"]),
            verbose = 1,
            callbacks = callbacks,
            validation_data = (self.x_val, self.y_val),
            shuffle = bool(self.training_process_hyperparameters["shuffle"])
        )

    def save_model(self, model_name: str = "final") -> None:
        self.model.save_weights(self.output_dir + model_name)

    def load_model(self, model_name: str = "final") -> None:
        latest_checkpoint = tf.train.latest_checkpoint(self.output_dir + model_name)

        self.__load_hyperparameters() 
        self.create_model()
        self.model.load_weights(latest_checkpoint)

    def __load_hyperparameters(self):
        pass

    def evaluate_model(self, x: List[str], y: List[str]):
        val_loss, val_acc = self.model.evaluate(self.x_val,  self.y_val, verbose=2)
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
