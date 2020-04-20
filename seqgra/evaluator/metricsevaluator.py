"""
MIT - CSAIL - Gifford Lab - seqgra

metrics evaluator: evaluates model using conventional performance metrics

calculates accuracy and loss for training, validation and test set

@author: Konstantin Krismer
"""
import pandas as pd

from seqgra.learner.learner import Learner
from seqgra.evaluator.evaluator import Evaluator


class MetricsEvaluator(Evaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("metrics", learner, output_dir)

    def evaluate_model(self, set_name: str = "test") -> None:
        # load data
        set_file: str = self.learner.get_examples_file(set_name)
        x, y = self.learner.parse_data(set_file)

        metrics = self.learner.evaluate_model(x=x, y=y)

        self.__save_results(metrics, set_name)

    def __save_results(self, results, set_name: str) -> None:
        if results is None:
            df = pd.DataFrame([], columns=["set", "metric", "value"])
        else:
            df = pd.DataFrame([[set_name, "loss", results["loss"]],
                               [set_name, "accuracy", results["accuracy"]]],
                              columns=["set", "metric", "value"])

        df.to_csv(self.output_dir + set_name + "-metrics.txt", sep="\t",
                  index=False)
