"""
MIT - CSAIL - Gifford Lab - seqgra

predict evaluator: writes model predictions of all examples in set to file

@author: Konstantin Krismer
"""
from typing import Any, List

import pandas as pd

import seqgra.constants as c
from seqgra.learner import Learner
from seqgra.evaluator import Evaluator


class PredictEvaluator(Evaluator):
    def __init__(self, learner: Learner, output_dir: str,
                 silent: bool = False) -> None:
        super().__init__(c.EvaluatorID.PREDICT, "Prediction",
                         learner, output_dir, silent=silent)

    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        encoded_y = self.learner.encode_y(y)
        y_hat = self.learner.predict(x=x)

        return (encoded_y, y_hat)

    def _save_results(self, results, set_name: str = "test",
                      suppress_plots: bool = False) -> None:
        if results is not None:
            y_df = pd.DataFrame(
                results[0], columns=["y_" + s
                                     for s in self.learner.definition.labels])
            y_hat_df = pd.DataFrame(
                results[1],
                columns=["y_hat_" + s
                         for s in self.learner.definition.labels])
            df = pd.concat([y_df, y_hat_df], axis=1)
            df.to_csv(self.output_dir + set_name + "-y-hat.txt", sep="\t",
                      index=False)
