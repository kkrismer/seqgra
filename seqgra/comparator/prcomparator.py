"""
MIT - CSAIL - Gifford Lab - seqgra

ROC comparator: creates ROC curves of different models

@author: Konstantin Krismer
"""
from typing import List, Optional
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import seqgra.constants as c
from seqgra.comparator import Comparator


class PRComparator(Comparator):
    def __init__(self, analysis_name: str, output_dir: str,
                 model_labels: Optional[List[str]] = None,
                 silent: bool = False) -> None:
        super().__init__(c.ComparatorID.PR, "PR curve", analysis_name,
                         output_dir, model_labels, silent)

    def compare_models(self, grammar_ids: Optional[List[str]] = None,
                       model_ids: Optional[List[str]] = None,
                       set_names: List[str] = None) -> None:
        if not set_names:
            set_names = ["test"]

        precision: List[List[float]] = list()
        recall: List[List[float]] = list()
        pr_auc: List[float] = list()
        valid_labels: List[str] = list()
        if len(grammar_ids) == 1 and len(set_names) == 1:
            for model_id in model_ids:
                predict_file_name: str = self.evaluation_dir + \
                    grammar_ids[0] + "/" + \
                    model_id + "/" + c.EvaluatorID.PREDICT + \
                    "/test-y-hat.txt"
                if os.path.isfile(predict_file_name):
                    df = pd.read_csv(predict_file_name, sep="\t")
                    num_labels: int = int(len(df.columns) / 2)
                    y_df = df.iloc[:, 0:num_labels]
                    y_hat_df = df.iloc[:, num_labels:len(df.columns)]

                    current_precision, current_recall, current_auc = self.create_single_precision_recall_curve(
                        y_df.values, y_hat_df.values)
                    precision.append(current_precision)
                    recall.append(current_recall)
                    pr_auc.append(current_auc)
                    valid_labels.append(model_id)
                else:
                    self.logger.warning("file does not exist: %s",
                                        predict_file_name)
        elif len(model_ids) == 1 and len(set_names) == 1:
            for grammar_id in grammar_ids:
                predict_file_name: str = self.evaluation_dir + \
                    grammar_id + "/" + \
                    model_ids[0] + "/" + c.EvaluatorID.PREDICT + \
                    "/test-y-hat.txt"
                if os.path.isfile(predict_file_name):
                    df = pd.read_csv(predict_file_name, sep="\t")
                    num_labels: int = int(len(df.columns) / 2)
                    y_df = df.iloc[:, 0:num_labels]
                    y_hat_df = df.iloc[:, num_labels:len(df.columns)]

                    current_precision, current_recall, current_auc = self.create_single_precision_recall_curve(
                        y_df.values, y_hat_df.values)
                    precision.append(current_precision)
                    recall.append(current_recall)
                    pr_auc.append(current_auc)
                    valid_labels.append(grammar_id)
                else:
                    self.logger.warning("file does not exist: %s",
                                        predict_file_name)
        else:
            # TODO
            pass

        if not self.model_labels or len(self.model_labels) != len(valid_labels):
            self.model_labels = valid_labels

        self.create_precision_recall_curve(precision, recall, pr_auc,
                                           self.model_labels,
                                           self.output_dir + set_names[0] +
                                           "-pr-curve.pdf")

    def create_single_precision_recall_curve(self, y_true, y_hat) -> None:
        """Create precision-recall curve.

        Plots PR curves for each class label, including micro-average and
        iso-F1 curves. Saves plot as PDF in `file_name`.

        Arguments:
            y_true (array): TODO ; shape = [n_samples, n_classes]
            y_hat (array): TODO ; shape = [n_samples, n_classes]
        """
        # Compute micro-average PR curve and PR curve area
        precision, recall, _ = precision_recall_curve(
            y_true.ravel(), y_hat.ravel())
        pr_auc = average_precision_score(y_true, y_hat, average="micro")

        return precision, recall, pr_auc

    def create_precision_recall_curve(self, precision: List[List[float]],
                                      recall: List[List[float]],
                                      pr_auc: List[float],
                                      model_labels: List[str],
                                      file_name: str) -> None:
        n_lines = len(model_labels)
        # Plot all ROC curves
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

        for i in range(n_lines):
            line, = plt.plot(recall[i], precision[i], linewidth=2)
            lines.append(line)
            labels.append(
                "{0} (area = {1:0.2f})"
                "".format(model_labels[i], pr_auc[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend(lines, labels, bbox_to_anchor=(1.04, 1),
                   loc="upper left", prop=dict(size=14))
        plt.savefig(file_name, bbox_inches="tight")
