"""
@author: Konstantin Krismer
"""

from abc import ABC, abstractmethod

class Learner(ABC):
    @abstractmethod
    def parse_data(self, x_train, y_train, x_val, y_val, x_test, y_test) -> None:
        """ This is the forward calculation from x to y
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        pass

    @abstractmethod
    def predict(self):
        """ This is the forward calculation from x to y
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        pass

    @abstractmethod
    def train_model(self):
        """Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Returns:
            train_op: The Op for training.
        """
        pass

    @abstractmethod
    def evaluate_model(self):
        """Evaluate the quality of the logits at predicting the label.
        Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
            A scalar int32 tensor with the number of examples (out of batch_size)
            that were predicted correctly.
        """
        pass
