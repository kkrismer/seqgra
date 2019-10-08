"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all learners
- abstract class for multi-class classification learners, i.e., learners
  for data where the class labels are mututally exclusive
- abstract class for multi-label classification learners, i.e., learners
  for data where the class labels are not mututally exclusive
- abstract class for multiple regression learners, i.e., learners with 
  multiple independent variables and one dependent variable
- abstract class for multivariate regression learners, i.e., learners with 
  multiple independent variables and multiple dependent variables

@author: Konstantin Krismer
"""

from abc import ABC, abstractmethod
from typing import List

class Learner(ABC):
    @abstractmethod
    def parse_data(self, x_train: List[str], y_train: List[str], x_val: List[str], y_val: List[str]) -> None:
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
    def save_model(self, model_name: str):
        pass

class MultiClassClassificationLearner(Learner):
    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[str]):
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

class MultiLabelClassificationLearner(Learner):
    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[List[str]]):
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

class MultipleRegressionLearner(Learner):
    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[float]):
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

class MultivariateRegressionLearner(Learner):
    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[List[float]]):
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
