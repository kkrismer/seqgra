"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from typing import List, Any
import tensorflow as tf
import os

from seqgra.learner.dna import DNAMultiClassClassificationLearner

class TensorFlowKerasSequentialLearner(DNAMultiClassClassificationLearner):
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)
        self.model = None
        self.hyperparameters = None

    def create_model(self) -> None:
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(150, 4)),
            tf.keras.layers.Dense(2, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy", # binary_crossentropy for multi-label
            metrics=["accuracy"]
        )

    def print_model_summary(self):
        self.model.summary()

    def train_model(self):
        if self.x_train is None or self.y_train is None or self.x_val is None or self.y_val is None:
            raise Exception("training and / or validation data has not been loaded")

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
        es_callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
            mode = "min", verbose = 1, patience = 2, min_delta = 1)

        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size = 32,
            epochs=10,
            verbose=1,
            callbacks=[cp_callback, tensorboard_callback, es_callback],
            validation_data=(self.x_val, self.y_val),
            shuffle=True
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
