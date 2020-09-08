"""
MIT - CSAIL - Gifford Lab - seqgra

ROC evaluator: creates ROC curves

@author: Konstantin Krismer
"""
from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

import seqgra.constants as c
from seqgra.learner import Learner
from seqgra.evaluator import Evaluator


class ROCEvaluator(Evaluator):
    def __init__(self, learner: Learner, output_dir: str,
                 silent: bool = False) -> None:
        super().__init__(c.EvaluatorID.ROC, "ROC curve", learner, output_dir,
                         silent=silent)

    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        encoded_y = self.learner.encode_y(y)
        y_hat = self.learner.predict(x)

        return (encoded_y, y_hat)

    def _save_results(self, results, set_name: str = "test",
                      suppress_plots: bool = False) -> None:
        if not suppress_plots:
            self.create_roc_curve(results[0], results[1],
                                  self.output_dir + set_name +
                                  "-roc-curve.pdf")

    def create_roc_curve(self, y_true, y_hat, file_name) -> None:
        """Create ROC curve.

        Plots ROC curves for each class label, including micro-average and
        macro-average. Saves plot as PDF in `file_name`.

        Arguments:
            y_true (array): TODO ; shape = [n_samples, n_classes]
            y_hat (array): TODO ; shape = [n_samples, n_classes]
            file_name (str): TODO
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(self.learner.definition.labels)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_hat[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true.ravel(), y_hat.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(7, 7))
        lines = []
        labels = []

        line, = plt.plot(fpr["micro"], tpr["micro"],
                         color="gold", linestyle=":", linewidth=2)
        lines.append(line)
        labels.append("micro-average (area = {0:0.2f})"
                      "".format(roc_auc["micro"]))

        line, = plt.plot(fpr["macro"], tpr["macro"],
                         color="darkorange", linestyle=":", linewidth=2)
        lines.append(line)
        labels.append("macro-average (area = {0:0.2f})"
                      "".format(roc_auc["macro"]))

        for i in range(n_classes):
            line, = plt.plot(fpr[i], tpr[i], linewidth=2)
            lines.append(line)
            labels.append(
                "condition {0} (area = {1:0.2f})"
                "".format(self.learner.definition.labels[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], "k--", linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(lines, labels, bbox_to_anchor=(1.04, 1),
                   loc="upper left", prop=dict(size=14))
        plt.savefig(file_name, bbox_inches="tight")
