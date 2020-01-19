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

from seqgra.learner.dna import DNAMultiLabelClassificationLearner
from seqgra.parser.modelparser import ModelParser
from seqgra.learner.kerashelper import KerasHelper


class KerasSequentialMultiLabelClassificationLearner(DNAMultiLabelClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)
        self.model = None

    def create_model(self) -> None:
        self.set_seed()
        self.model = tf.keras.Sequential(
            [KerasHelper.get_keras_layer(operation)
             for operation
             in self.architecture.operations])

        self.model.compile(
            optimizer=KerasHelper.get_optimizer(self.optimizer_hyperparameters),
            # use categorical_crossentropy for multi-class and 
            # binary_crossentropy for multi-label
            loss=KerasHelper.get_loss(self.loss_hyperparameters),
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
            session_file.write("seqgra package version: " +
                pkg_resources.require("seqgra")[0].version + "\n")
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
