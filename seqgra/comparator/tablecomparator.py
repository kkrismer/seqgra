"""
MIT - CSAIL - Gifford Lab - seqgra
"""
from typing import List, Optional
import os

import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score

import seqgra.constants as c
from seqgra.comparator import Comparator
from seqgra import Metrics


class TableComparator(Comparator):
    def __init__(self, analysis_name: str, output_dir: str,
                 model_labels: Optional[List[str]] = None) -> None:
        super().__init__(c.ComparatorID.TABLE, "Table", analysis_name,
                         output_dir, model_labels)

    def compare_models(self, grammar_ids: Optional[List[str]] = None,
                       model_ids: Optional[List[str]] = None,
                       set_names: List[str] = None) -> None:
        if not set_names:
            set_names = ["test"]

        grammar_id_column: List[str] = list()
        model_id_column: List[str] = list()
        set_name_column: List[str] = list()
        d_training_set_size_column: List[int] = list()
        d_validation_set_size_column: List[int] = list()
        d_test_set_size_column: List[int] = list()
        m_last_epoch_completed_column: List[int] = list()
        e_num_labels_column: List[int] = list()
        e_metrics_loss_column: List[float] = list()
        e_metrics_accuracy_column: List[float] = list()
        e_roc_micro_auc_column: List[float] = list()
        e_roc_macro_auc_column: List[float] = list()
        e_pr_micro_auc_column: List[float] = list()
        e_pr_macro_auc_column: List[float] = list()

        for grammar_id in grammar_ids:
            training_set_size: int = self.get_set_size(grammar_id, "training")
            validation_set_size: int = self.get_set_size(grammar_id,
                                                         "validation")
            test_set_size: int = self.get_set_size(grammar_id, "test")

            for model_id in model_ids:
                last_epoch_completed: int = self.get_last_epoch_completed(
                    grammar_id, model_id)
                for set_name in set_names:
                    num_labels: int = self.get_num_labels(grammar_id,
                                                          model_id, set_name)
                    metrics_loss, metrics_accuracy = self.get_metrics(
                        grammar_id, model_id, set_name)
                    roc_micro_auc, roc_macro_auc = self.get_roc_auc(
                        grammar_id, model_id, set_name)
                    pr_micro_auc, pr_macro_auc = self.get_pr_auc(
                        grammar_id, model_id, set_name)

                    grammar_id_column.append(grammar_id)
                    model_id_column.append(model_id)
                    set_name_column.append(set_name)
                    d_training_set_size_column.append(training_set_size)
                    d_validation_set_size_column.append(validation_set_size)
                    d_test_set_size_column.append(test_set_size)
                    m_last_epoch_completed_column.append(last_epoch_completed)
                    e_num_labels_column.append(num_labels)
                    e_metrics_loss_column.append(metrics_loss)
                    e_metrics_accuracy_column.append(metrics_accuracy)
                    e_roc_micro_auc_column.append(roc_micro_auc)
                    e_roc_macro_auc_column.append(roc_macro_auc)
                    e_pr_micro_auc_column.append(pr_micro_auc)
                    e_pr_macro_auc_column.append(pr_macro_auc)

        df = pd.DataFrame(
            {"grammar_id": grammar_id_column,
             "model_id": model_id_column,
             "set_name": set_name_column,
             "training_set_size": d_training_set_size_column,
             "validation_set_size": d_validation_set_size_column,
             "test_set_size": d_test_set_size_column,
             "last_epoch_completed": m_last_epoch_completed_column,
             "num_labels": e_num_labels_column,
             "metrics_loss": e_metrics_loss_column,
             "metrics_accuracy": e_metrics_accuracy_column,
             "roc_micro_auc": e_roc_micro_auc_column,
             "roc_macro_auc": e_roc_macro_auc_column,
             "pr_micro_auc": e_pr_micro_auc_column,
             "pr_macro_auc": e_pr_macro_auc_column})
        df.to_csv(self.output_dir + "table.txt", sep="\t", index=False)

    def get_set_size(self, grammar_id: str, set_name: str) -> int:
        set_file_name: str = self.data_dir + grammar_id + "/" + \
            set_name + ".txt"
        i: int = -1  # do not count header
        if os.path.isfile(set_file_name):
            with open(set_file_name) as f:
                for line in f:
                    i += 1

        return i

    def get_last_epoch_completed(self, grammar_id: str, model_id: str) -> int:
        last_epoch_file_name: str = self.model_dir + grammar_id + "/" + \
            model_id + "/last-epoch-completed.txt"
        last_epoch: int = -1
        if os.path.isfile(last_epoch_file_name):
            with open(last_epoch_file_name) as f:
                last_epoch = int(f.readline().strip())

        return last_epoch

    def get_num_labels(self, grammar_id: str, model_id: str,
                       set_name: str) -> int:
        predict_file_name: str = self.evaluation_dir + \
            grammar_id + "/" + \
            model_id + "/" + c.EvaluatorID.PREDICT + \
            "/" + set_name + "-y-hat.txt"
        if os.path.isfile(predict_file_name):
            df = pd.read_csv(predict_file_name, sep="\t")
            return int(len(df.columns) / 2)
        else:
            return -1

    def get_metrics(self, grammar_id: str, model_id: str,
                    set_name: str) -> Metrics:
        metrics_file_name: str = self.evaluation_dir + \
            grammar_id + "/" + \
            model_id + "/" + c.EvaluatorID.METRICS + \
            "/" + set_name + "-metrics.txt"
        if os.path.isfile(metrics_file_name):
            df = pd.read_csv(metrics_file_name, sep="\t")

            loss = df[df["metric"] == "loss"]
            accuracy = df[df["metric"] == "accuracy"]

            return Metrics(loss.iloc[0]["value"], accuracy.iloc[0]["value"])
        else:
            return Metrics(-1, -1)

    def get_roc_auc(self, grammar_id: str, model_id: str,
                    set_name: str) -> [float, float]:
        predict_file_name: str = self.evaluation_dir + \
            grammar_id + "/" + \
            model_id + "/" + c.EvaluatorID.PREDICT + \
            "/" + set_name + "-y-hat.txt"
        if os.path.isfile(predict_file_name):
            df = pd.read_csv(predict_file_name, sep="\t")
            num_labels: int = int(len(df.columns) / 2)
            y_df = df.iloc[:, 0:num_labels]
            y_hat_df = df.iloc[:, num_labels:len(df.columns)]

            micro_fpr, micro_tpr, _ = roc_curve(y_df.values.ravel(),
                                                y_hat_df.values.ravel())
            roc_micro_auc = auc(micro_fpr, micro_tpr)

            fpr = dict()
            tpr = dict()
            for i in range(num_labels):
                fpr[i], tpr[i], _ = roc_curve(y_df.values[:, i],
                                              y_hat_df.values[:, i])

            all_fpr = np.unique(np.concatenate([fpr[i]
                                                for i in range(num_labels)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(num_labels):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            mean_tpr /= num_labels

            roc_macro_auc = auc(all_fpr, mean_tpr)

            return (roc_micro_auc, roc_macro_auc)
        else:
            return (-1, -1)

    def get_pr_auc(self, grammar_id: str, model_id: str,
                   set_name: str) -> [float, float]:
        predict_file_name: str = self.evaluation_dir + \
            grammar_id + "/" + \
            model_id + "/" + c.EvaluatorID.PREDICT + \
            "/" + set_name + "-y-hat.txt"
        if os.path.isfile(predict_file_name):
            df = pd.read_csv(predict_file_name, sep="\t")
            num_labels: int = int(len(df.columns) / 2)
            y_df = df.iloc[:, 0:num_labels]
            y_hat_df = df.iloc[:, num_labels:len(df.columns)]

            pr_micro_auc = average_precision_score(y_df.values,
                                                   y_hat_df.values,
                                                   average="micro")
            pr_macro_auc = average_precision_score(y_df.values,
                                                   y_hat_df.values,
                                                   average="macro")

            return (pr_micro_auc, pr_macro_auc)
        else:
            return (-1, -1)
