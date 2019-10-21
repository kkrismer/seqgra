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
import random
from abc import ABC, abstractmethod
from typing import List, Any, Dict

import numpy as np

from seqgra.parser.modelparser import ModelParser
from seqgra.model.model.architecture import Architecture
from seqgra.model.model.operation import Operation
from seqgra.model.model.metric import Metric

class Learner(ABC):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        self._parser: ModelParser = parser
        self.__parse_config()
        output_dir = output_dir.replace("\\", "/")
        if not output_dir.endswith("/"):
            output_dir += "/"
        self.output_dir = output_dir
        self.__prepare_output_dir()

    def __str__(self):
        str_rep = ["seqgra model configuration:\n",
            "\tID: ", self.id, "\n",
            "\tLabel: ", self.label, "\n",
            "\tDescription:\n"]
        if self.description:
            str_rep += ["\t", self.description, "\n"]
        str_rep += ["\tLibrary: ", self.library, "\n",
            "\tLearner type: ", self.learner_type, "\t",
            "\tLearner implementation", self.learner_implementation]

        str_rep += ["\tMetrics:\n"]
        for metric in self.metrics:
            str_rep += ["\t\t" + s + "\n" for s in str(metric).splitlines()]

        str_rep += ["\t" + s + "\n" for s in str(self.architecture).splitlines()]
        
        str_rep += ["\tLoss hyperparameters:\n", "\t\t",
            str(self.loss_hyperparameters), "\n"]
        str_rep += ["\tOptimizer hyperparameters:\n", "\t\t",
            str(self.optimizer_hyperparameters), "\n"]
        str_rep += ["\tTraining process hyperparameters:\n", "\t\t",
            str(self.training_process_hyperparameters), "\n"]

        return "".join(str_rep)

    def __parse_config(self):
        self.id: str = self._parser.get_id()
        self.label: str = self._parser.get_label()
        self.description: str = self._parser.get_description()
        self.library: str = self._parser.get_library()
        self.learner_type: str = self._parser.get_learner_type()
        self.learner_implementation: str = self._parser.get_learner_implementation()
        self.seed: int = self._parser.get_seed()
        self.metrics: List[Metric] = self._parser.get_metrics()
        self.architecture: Architecture = self._parser.get_architecture()
        self.loss_hyperparameters: Dict[str, str] = self._parser.get_loss_hyperparameters()
        self.optimizer_hyperparameters: Dict[str, str] = self._parser.get_optimizer_hyperparameters()
        self.training_process_hyperparameters: Dict[str, str] = self._parser.get_training_process_hyperparameters()

    def __set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

    @abstractmethod
    def parse_data(self, training_file: str, validation_file: str) -> None:
        pass

    @abstractmethod
    def create_model(self) -> None:
        pass

    def train_model(self) -> None:
        self.__set_seed()
        self.__train_model()
        
    @abstractmethod
    def __train_model(self) -> None:
        pass

    @abstractmethod
    def save_model(self, model_name: str = "final"):
        pass

    @abstractmethod
    def load_model(self, model_name: str = "final"):
        pass

    @abstractmethod
    def print_model_summary(self) -> None:
        pass

    def __prepare_output_dir(self) -> None:
        if os.path.exists(self.output_dir):
            if os.path.isdir(self.output_dir):
                if len(os.listdir(self.output_dir)) > 0:
                    pass
                    # raise Exception("output directory non-empty")
            else:
                raise Exception("output directory cannot be created (file with same name exists)")
        else:    
            os.makedirs(self.output_dir)

    @abstractmethod
    def predict(self, x: Any, encode: bool = True):
        pass

class MultiClassClassificationLearner(Learner):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

        if self.learner_type != "multi-class classification learner":
            raise Exception("model definition must specify multi-class classification learner type, but learner type is '" + self.learner_type, "'")

    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[str]):
        pass

class MultiLabelClassificationLearner(Learner):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

        if self.learner_type != "multi-label classification learner":
            raise Exception("model definition must specify multi-label classification learner type, but learner type is '" + self.learner_type, "'")

    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[List[str]]):
        pass

class MultipleRegressionLearner(Learner):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

        if self.learner_type != "multiple regression learner":
            raise Exception("model definition must specify multiple regression learner type, but learner type is '" + self.learner_type, "'")

    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[float]):
        pass

class MultivariateRegressionLearner(Learner):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

        if self.learner_type != "multivariate regression learner":
            raise Exception("model definition must specify multivariate regression learner type, but learner type is '" + self.learner_type, "'")

    @abstractmethod
    def evaluate_model(self, x: List[str], y: List[List[float]]):
        pass
