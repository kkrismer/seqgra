"""
MIT - CSAIL - Gifford Lab - seqgra
"""
from typing import List, Optional
import os

import numpy as np
import pandas as pd

import seqgra.constants as c
from seqgra.comparator import Comparator


class FIETableComparator(Comparator):
    def __init__(self, analysis_id: str, output_dir: str,
                 model_labels: Optional[List[str]] = None,
                 silent: bool = False) -> None:
        super().__init__(c.ComparatorID.FEATURE_IMPORTANCE_EVALUATOR_TABLE,
                         "Feature Importance Evaluator Table",
                         analysis_id, output_dir, model_labels, silent)

    def compare_models(self, grammar_ids: Optional[List[str]] = None,
                       model_ids: Optional[List[str]] = None,
                       set_names: List[str] = None) -> None:
        if not set_names:
            set_names = ["test"]

        grammar_id_column: List[str] = list()
        model_id_column: List[str] = list()
        set_name_column: List[str] = list()
        evaluator_id_column: List[str] = list()
        thresholded_column: List[bool] = list()
        label_column: List[str] = list()
        precision_column: List[float] = list()
        recall_column: List[float] = list()
        specificity_column: List[float] = list()
        f1_column: List[float] = list()
        n_column: List[int] = list()

        for grammar_id in grammar_ids:
            for model_id in model_ids:
                for set_name in set_names:
                    labels: List[str] = self.get_labels(grammar_id, model_id,
                                                        set_name)
                    if labels is None:
                        labels = list()
                    for evaluator_id in c.EvaluatorID.FEATURE_IMPORTANCE_EVALUATORS:
                        no_valid_file_name: str = self.evaluation_dir + \
                            grammar_id + "/" + model_id + "/" + \
                            evaluator_id + "/" + set_name + \
                            "-no-valid-examples.txt"
                        for thresholded in [True, False]:
                            if thresholded:
                                statistics_file_name: str = self.evaluation_dir + \
                                    grammar_id + "/" + model_id + "/" + \
                                    evaluator_id + "/" + set_name + \
                                    "-statistics-thresholded.txt"
                            else:
                                statistics_file_name: str = self.evaluation_dir + \
                                    grammar_id + "/" + model_id + "/" + \
                                    evaluator_id + "/" + set_name + \
                                    "-statistics.txt"

                            if os.path.isfile(statistics_file_name):
                                df = pd.read_csv(statistics_file_name,
                                                 sep="\t")

                                current_labels: List[str] = labels.copy()
                                for _, row in df.iterrows():
                                    label: str = row["label"].strip()
                                    if label in current_labels:
                                        current_labels.remove(label)
                                    grammar_id_column.append(grammar_id)
                                    model_id_column.append(model_id)
                                    set_name_column.append(set_name)
                                    evaluator_id_column.append(evaluator_id)
                                    thresholded_column.append(thresholded)
                                    label_column.append(label)
                                    precision_column.append(row["precision"])
                                    recall_column.append(row["recall"])
                                    specificity_column.append(
                                        row["specificity"])
                                    f1_column.append(row["f1"])
                                    n_column.append(row["n"])

                                # add NA for missing labels (labels without
                                # valid examples)
                                if current_labels:
                                    for label in current_labels:
                                        grammar_id_column.append(grammar_id)
                                        model_id_column.append(model_id)
                                        set_name_column.append(set_name)
                                        evaluator_id_column.append(
                                            evaluator_id)
                                        thresholded_column.append(thresholded)
                                        label_column.append(label)
                                        precision_column.append(np.nan)
                                        recall_column.append(np.nan)
                                        specificity_column.append(np.nan)
                                        f1_column.append(np.nan)
                                        n_column.append(np.nan)
                            elif os.path.isfile(no_valid_file_name):
                                # add NA for all labels (labels without
                                # valid examples)
                                if labels:
                                    for label in labels:
                                        grammar_id_column.append(grammar_id)
                                        model_id_column.append(model_id)
                                        set_name_column.append(set_name)
                                        evaluator_id_column.append(
                                            evaluator_id)
                                        thresholded_column.append(thresholded)
                                        label_column.append(label)
                                        precision_column.append(np.nan)
                                        recall_column.append(np.nan)
                                        specificity_column.append(np.nan)
                                        f1_column.append(np.nan)
                                        n_column.append(np.nan)
                            elif (thresholded and
                                  evaluator_id == c.EvaluatorID.SIS) or \
                                    evaluator_id in c.EvaluatorID.CORE_FEATURE_IMPORTANCE_EVALUATORS:
                                self.logger.warning("file does not exist: %s",
                                                    statistics_file_name)

        df = pd.DataFrame(
            {"grammar_id": grammar_id_column,
             "model_id": model_id_column,
             "set_name": set_name_column,
             "evaluator_id": evaluator_id_column,
             "thresholded": thresholded_column,
             "label": label_column,
             "precision": precision_column,
             "recall": recall_column,
             "specificity": specificity_column,
             "f1": f1_column,
             "n": n_column})
        df.to_csv(self.output_dir + "fie-table.txt", sep="\t", index=False)
