"""MIT - CSAIL - Gifford Lab - seqgra

TensorFlow Keras learner helper class

@author: Konstantin Krismer
"""
import logging
import os
import random
import sys
from typing import Any, List

import numpy as np
import pkg_resources

from seqgra.learner import Learner
from seqgra.model import DataDefinition
from seqgra.model.model import Architecture
from seqgra.parser import DataDefinitionParser
from seqgra.parser import XMLDataDefinitionParser


class BayesOptimalHelper:
    @staticmethod
    def create_model(learner: Learner) -> None:
        arch: Architecture = learner.definition.architecture
        if arch.external_model_format != "data-definition":
            raise Exception("invalid external model format: " +
                            arch.external_model_format)
        path: str = arch.external_model_path
        if os.path.isfile(path):
            with open(path, "r") as data_definition_file:
                data_config: str = data_definition_file.read()
            data_def_parser: DataDefinitionParser = XMLDataDefinitionParser(
                data_config)

            data_definition: DataDefinition = data_def_parser.get_data_definition()
            learner.model = (data_definition.conditions,
                             data_definition.sequence_elements)
        else:
            raise Exception("data definition file does not exist: " + path)

    @staticmethod
    def print_model_summary(learner: Learner):
        if learner.model:
            for condition in learner.model[0]:
                print(condition)
            for sequence_element in learner.model[1]:
                print(sequence_element)
        else:
            print("uninitialized model")

    @staticmethod
    def set_seed(learner: Learner) -> None:
        random.seed(learner.definition.seed)
        np.random.seed(learner.definition.seed)

    @staticmethod
    def save_model(learner: Learner, model_name: str = "") -> None:
        pass

    @staticmethod
    def write_session_info(learner: Learner) -> None:
        with open(learner.output_dir + "session-info.txt", "w") as session_file:
            session_file.write(
                "seqgra package version: " +
                pkg_resources.require("seqgra")[0].version + "\n")
            session_file.write("NumPy version: " + np.version.version + "\n")
            session_file.write("Python version: " + sys.version + "\n")

    @staticmethod
    def load_model(learner: Learner, model_name: str = "") -> None:
        pass

    @staticmethod
    def predict(learner: Learner, x: Any, encode: bool = True):
        """ This is the forward calculation from x to y
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        if encode:
            x = learner.encode_x(x)

        return 0.5

    @staticmethod
    def evaluate_model(learner: Learner, x: List[str], y: List[str]):
        # one hot encode input and labels
        encoded_x = learner.encode_x(x)
        encoded_y = learner.encode_y(y)

        loss, accuracy = learner.model.evaluate(encoded_x, encoded_y,
                                                verbose=0)
        return {"loss": loss, "accuracy": accuracy}
