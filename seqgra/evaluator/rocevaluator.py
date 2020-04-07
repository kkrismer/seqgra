"""
MIT - CSAIL - Gifford Lab - seqgra

ROC evaluator: creates ROC curves

@author: Konstantin Krismer
"""
import os

import pandas as pd

from seqgra.learner.learner import Learner
from seqgra.evaluator.evaluator import Evaluator


class ROCEvaluator(Evaluator):
    def __init__(self, learner: Learner, data_dir: str,
                 output_dir: str) -> None:
        super().__init__("roc", learner, data_dir, output_dir)

    def evaluate_model(self, set_name: str = "test") -> None:
        # load data
        set_file: str = Evaluator.get_valid_file(self.data_dir + "/" +
                                                 set_name + ".txt")
        x, y = self.learner.parse_data(set_file)
        encoded_y = self.learner.encode_y(y)
        y_hat = self.learner.predict(x)

        self.learner.create_roc_curve(encoded_y, y_hat,
                                      self.output_dir + set_name +
                                      "-roc-curve.pdf")
