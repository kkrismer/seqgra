"""
MIT - CSAIL - Gifford Lab - seqgra

metrics evaluator: evaluates model using conventional performance metrics

calculates accuracy and loss for training, validation and test set

@author: Konstantin Krismer
"""
from typing import Any, List

import pandas as pd

import seqgra.constants as c
from seqgra.learner import Learner
from seqgra.evaluator import Evaluator


class MetricsEvaluator(Evaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.METRICS, "Accuracy and loss metrics",
                         learner, output_dir)

    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        return self.learner.evaluate_model(x=x, y=y)

    def _save_results(self, results, set_name: str = "test",
                      suppress_plots: bool = False) -> None:
        if results is None:
            df = pd.DataFrame([], columns=["set", "metric", "value"])
        else:
            df = pd.DataFrame([[set_name, "loss", results["loss"]],
                               [set_name, "accuracy", results["accuracy"]]],
                              columns=["set", "metric", "value"])

        df.to_csv(self.output_dir + set_name + "-metrics.txt", sep="\t",
                  index=False)
