"""
MIT - CSAIL - Gifford Lab - seqgra

predict evaluator: writes model predictions of all examples in set to file

@author: Konstantin Krismer
"""
import pandas as pd

from seqgra.learner.learner import Learner
from seqgra.evaluator.evaluator import Evaluator


class PredictEvaluator(Evaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("predict", learner, output_dir)

    def evaluate_model(self, set_name: str = "test") -> None:
        # load data
        set_file: str = self.learner.get_examples_file(set_name)
        x, _ = self.learner.parse_data(set_file)

        y_hat = self.learner.predict(x)
        self.__save_results(y_hat, set_name)

    def __save_results(self, results, set_name: str) -> None:
        if results is None:
            df = pd.DataFrame([], columns=self.learner.labels)
        else:
            df = pd.DataFrame(results, columns=self.learner.labels)

        df.to_csv(self.output_dir + set_name + "-y-hat.txt", sep="\t",
                  index=False)
