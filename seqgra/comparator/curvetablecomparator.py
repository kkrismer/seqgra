"""
MIT - CSAIL - Gifford Lab - seqgra
"""
from typing import List, Optional
import os

import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score

import seqgra.constants as c
from seqgra.comparator import Comparator


class CurveTableComparator(Comparator):
    def __init__(self, analysis_name: str, output_dir: str,
                 model_labels: Optional[List[str]] = None,
                 silent: bool = False) -> None:
        super().__init__(c.ComparatorID.TABLE, "ROC and PR AUC Table",
                         analysis_name,
                         output_dir, model_labels, silent)

    def compare_models(self, grammar_ids: Optional[List[str]] = None,
                       model_ids: Optional[List[str]] = None,
                       set_names: List[str] = None) -> None:
        if not set_names:
            set_names = ["test"]

        grammar_id_column: List[str] = list()
        model_id_column: List[str] = list()
        set_name_column: List[str] = list()
        evaluator_id_column: List[str] = list()
        label_column: List[str] = list()
        auc_column: List[float] = list()
        n_column: List[int] = list()

        for grammar_id in grammar_ids:
            for model_id in model_ids:
                for set_name in set_names:
                    predict_file_name: str = self.evaluation_dir + \
                        grammar_id + "/" + \
                        model_id + "/" + c.EvaluatorID.PREDICT + \
                        "/" + set_name + "-y-hat.txt"

                    if os.path.isfile(predict_file_name):
                        df = pd.read_csv(predict_file_name, sep="\t")
                        num_labels: int = int(len(df.columns) / 2)
                        labels: List[str] = list(df.columns)[:num_labels]
                        labels = [label.replace("y_", "") for label in labels]
                        y_df = df.iloc[:, 0:num_labels]
                        y_hat_df = df.iloc[:, num_labels:len(df.columns)]

                        for evaluator_id in [c.EvaluatorID.ROC,
                                             c.EvaluatorID.PR]:
                            for i, label in enumerate(labels):
                                y = y_df.iloc[:, i].values
                                y_hat = y_hat_df.iloc[:, i].values
                                grammar_id_column.append(grammar_id)
                                model_id_column.append(model_id)
                                set_name_column.append(set_name)
                                evaluator_id_column.append(evaluator_id)
                                label_column.append(label.strip())
                                auc_column.append(self.get_per_label_auc(
                                    evaluator_id, y, y_hat))
                                n_column.append(sum(y))

        df = pd.DataFrame(
            {"grammar_id": grammar_id_column,
             "model_id": model_id_column,
             "set_name": set_name_column,
             "evaluator_id": evaluator_id_column,
             "label": label_column,
             "auc": auc_column,
             "n": n_column})
        df.to_csv(self.output_dir + "curve-table.txt", sep="\t", index=False)

    def get_per_label_auc(self, evaluator_id: str, y, y_hat) -> float:
        if evaluator_id == c.EvaluatorID.ROC:
            fpr, tpr, _ = roc_curve(y, y_hat)
            return auc(fpr, tpr)
        elif evaluator_id == c.EvaluatorID.PR:
            # average precision is AUC interpolated by constant segments,
            # which is more common than the AUC obtained by trapezoidal
            # interpolation (using `auc(precision, recall)`)
            return average_precision_score(y, y_hat)
