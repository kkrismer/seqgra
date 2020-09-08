"""
MIT - CSAIL - Gifford Lab - seqgra

PR evaluator: creates precision-recall curves

@author: Konstantin Krismer
"""
from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import seqgra.constants as c
from seqgra.learner import Learner
from seqgra.evaluator import Evaluator


class PREvaluator(Evaluator):
    def __init__(self, learner: Learner, output_dir: str,
                 silent: bool = False) -> None:
        super().__init__(c.EvaluatorID.PR, "Precision-recall curve",
                         learner, output_dir, silent=silent)

    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        encoded_y = self.learner.encode_y(y)
        y_hat = self.learner.predict(x)

        return (encoded_y, y_hat)

    def _save_results(self, results, set_name: str = "test",
                      suppress_plots: bool = False) -> None:
        if not suppress_plots:
            self.create_precision_recall_curve(results[0], results[1],
                                               self.output_dir + set_name +
                                               "-pr-curve.pdf")

    def create_precision_recall_curve(self, y_true, y_hat,
                                      file_name: str) -> None:
        """Create precision-recall curve.

        Plots PR curves for each class label, including micro-average and
        iso-F1 curves. Saves plot as PDF in `file_name`.

        Arguments:
            y_true (array): TODO ; shape = [n_samples, n_classes]
            y_hat (array): TODO ; shape = [n_samples, n_classes]
            file_name (str): TODO
        """
        precision = dict()
        recall = dict()
        average_precision = dict()
        n_classes = len(self.learner.definition.labels)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                                y_hat[:, i])
            average_precision[i] = average_precision_score(
                y_true[:, i], y_hat[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true.ravel(), y_hat.ravel())
        average_precision["micro"] = average_precision_score(y_true, y_hat,
                                                             average="micro")

        plt.figure(figsize=(7, 7))
        lines = []
        labels = []

        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.001, 1)
            y = f_score * x / (2 * x - f_score)
            line, = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate(r"$F_1 = {0:0.1f}$".format(
                f_score), xy=(0.89, y[45] + 0.02))

        lines.append(line)
        labels.append(r"iso-$F_1$ curves")
        line, = plt.plot(recall["micro"], precision["micro"],
                         linestyle=":", color="gold", linewidth=2)
        lines.append(line)
        labels.append("micro-average (area = {0:0.2f})"
                      "".format(average_precision["micro"]))

        for i in range(n_classes):
            line, = plt.plot(recall[i], precision[i], linewidth=2)
            lines.append(line)
            labels.append("condition {0} (area = {1:0.2f})"
                          "".format(self.learner.definition.labels[i],
                                    average_precision[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend(lines, labels, bbox_to_anchor=(1.04, 1),
                   loc="upper left", prop=dict(size=14))
        plt.savefig(file_name, bbox_inches="tight")
