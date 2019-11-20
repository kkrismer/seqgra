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
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.learner: Learner = learner
        self.data_dir: str = data_dir
        output_dir = output_dir.replace("\\", "/")
        if not output_dir.endswith("/"):
            output_dir += "/"
        self.output_dir = output_dir
        self.__prepare_output_dir()
        self.learner.set_seed()

    @abstractmethod
    def evaluate_model(self, set_name: str = "training") -> None:
        pass
    
    @abstractmethod
    def save_results(self, results, name: str) -> None:
        pass
    
    @abstractmethod
    def load_results(self, name: str):
        pass

    def write_session_info(self) -> None:
        with open(self.output_dir + "session-info.txt", "w") as session_file:
            session_file.write("seqgra package version: " + pkg_resources.require("seqgra")[0].version + "\n")
            session_file.write("NumPy version: " + np.version.version + "\n")
            session_file.write("Python version: " + sys.version + "\n")

    def __prepare_output_dir(self) -> None:
        if os.path.exists(self.output_dir):
            if not os.path.isdir(self.output_dir):
                raise Exception("output directory cannot be created (file with same name exists)")
        else:    
            os.makedirs(self.output_dir)
