"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Any, Dict

import pkg_resources
import numpy as np

from seqgra.learner.learner import Learner

class Evaluator(ABC):
    @abstractmethod
    def __init__(self, id: str, learner: Learner, data_dir: str, 
                 output_dir: str) -> None:
        self.id: str = id
        self.learner: Learner = learner
        self.data_dir: str = data_dir
        output_dir = output_dir.replace("\\", "/")
        if not output_dir.endswith("/"):
            output_dir += "/"
        self.output_dir = output_dir + id + "/"
        self.__prepare_output_dir()
        self.learner.set_seed()

    @abstractmethod
    def evaluate_model(self, set_name: str = "test") -> None:
        pass

    @staticmethod
    def get_valid_file(data_file: str) -> str:
        data_file = data_file.replace("\\", "/").replace("//", "/").strip()
        if os.path.isfile(data_file):
            return data_file
        else:
            raise Exception("file does not exist: " + data_file)

    def __prepare_output_dir(self) -> None:
        if os.path.exists(self.output_dir):
            if not os.path.isdir(self.output_dir):
                raise Exception("output directory cannot be created "
                                "(file with same name exists)")
        else:    
            os.makedirs(self.output_dir, exist_ok=True)
