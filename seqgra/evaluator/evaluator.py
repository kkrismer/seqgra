"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
import os
from abc import ABC, abstractmethod
from typing import List, Any, Dict

from seqgra.learner.learner import Learner

class Evaluator(ABC):
    @abstractmethod
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.learner: Learner = learner
        self.data_dir: str = data_dir
        output_dir = output_dir.replace("\\", "/")
        if not output_dir.endswith("/"):
            output_dir += "/"
        self.output_dir = output_dir
        self.__prepare_output_dir()

    @abstractmethod
    def evaluate(self, set = "training") -> None:
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
