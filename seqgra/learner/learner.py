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
import os

from abc import ABC, abstractmethod
from typing import List

class Learner(ABC):
    @abstractmethod
    def __init__(self, output_dir: str) -> None:
        output_dir = output_dir.replace("\\", "/")
        if not output_dir.endswith("/"):
            output_dir += "/"
        self.output_dir = output_dir
        self.__prepare_output_dir()

    @abstractmethod
    def parse_data(self, training_file: str, validation_file: str) -> None:
        pass

    @abstractmethod
    def create_model(self) -> None:
        pass

    @abstractmethod
    def train_model(self) -> None:
        pass

    @abstractmethod
    def save_model(self, model_name: str):
        pass

    @abstractmethod
    def print_model_summary(self) -> None:
        pass

    def __prepare_output_dir(self) -> None:
        if os.path.exists(self.output_dir):
            if os.path.isdir(self.output_dir):
                if len(os.listdir(self.output_dir)) > 0:
                    raise Exception("output directory non-empty")
            else:
                raise Exception("output directory cannot be created (file with same name exists)")
        else:    
            os.makedirs(self.output_dir)

    @abstractmethod
    def predict(self):
        pass

class MultiClassClassificationLearner(Learner):
    @abstractmethod
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)

    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[str]):
        pass

class MultiLabelClassificationLearner(Learner):
    @abstractmethod
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)

    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[List[str]]):
        pass

class MultipleRegressionLearner(Learner):
    @abstractmethod
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)

    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[float]):
        pass

class MultivariateRegressionLearner(Learner):
    @abstractmethod
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)

    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[List[float]]):
        pass
