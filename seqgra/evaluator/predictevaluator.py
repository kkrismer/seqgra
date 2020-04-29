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
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.PREDICT, "Prediction", learner, output_dir)

    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        return self.learner.predict(x)

    def _save_results(self, results, set_name: str = "test") -> None:
        if results is None:
            results = []

        df = pd.DataFrame(results, columns=self.learner.definition.labels)
        df.to_csv(self.output_dir + set_name + "-y-hat.txt", sep="\t",
                  index=False)
