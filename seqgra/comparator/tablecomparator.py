"""
MIT - CSAIL - Gifford Lab - seqgra
"""
from typing import List, Optional, Tuple
import os

import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score

import seqgra.constants as c
from seqgra.comparator import Comparator
from seqgra import Metrics
from seqgra.schema import DataSessionInfo, ModelSessionInfo


class TableComparator(Comparator):
    def __init__(self, analysis_id: str, output_dir: str,
                 model_labels: Optional[List[str]] = None,
                 silent: bool = False) -> None:
        super().__init__(c.ComparatorID.TABLE, "Table", analysis_id,
                         output_dir, model_labels, silent)

    def compare_models(self, grammar_ids: Optional[List[str]] = None,
                       model_ids: Optional[List[str]] = None,
                       set_names: List[str] = None) -> None:
        if not set_names:
            set_names = ["test"]

        grammar_id_column: List[str] = list()
        model_id_column: List[str] = list()
        set_name_column: List[str] = list()
        d_seqgra_version_column: List[str] = list()
        d_numpy_version_column: List[str] = list()
        d_python_version_column: List[str] = list()
        d_training_set_size_column: List[int] = list()
        d_validation_set_size_column: List[int] = list()
        d_test_set_size_column: List[int] = list()
        d_num_labels_column: List[int] = list()
        m_seqgra_version_column: List[str] = list()
        m_numpy_version_column: List[str] = list()
        m_python_version_column: List[str] = list()
        m_library_column: List[str] = list()
        m_library_version_column: List[str] = list()
        m_last_epoch_completed_column: List[int] = list()
        m_trainable_params_column: List[int] = list()
        m_non_trainable_params_column: List[int] = list()
        m_all_params_column: List[int] = list()
        e_metrics_loss_column: List[float] = list()
        e_metrics_accuracy_column: List[float] = list()
        e_roc_micro_auc_column: List[float] = list()
        e_roc_macro_auc_column: List[float] = list()
        e_pr_micro_auc_column: List[float] = list()
        e_pr_macro_auc_column: List[float] = list()

        for grammar_id in grammar_ids:
            data_session_info: DataSessionInfo = self.get_data_session_info(
                grammar_id)
            training_set_size: int = self.get_set_size(grammar_id, "training")
            validation_set_size: int = self.get_set_size(grammar_id,
                                                         "validation")
            test_set_size: int = self.get_set_size(grammar_id, "test")

            for model_id in model_ids:
                model_session_info: ModelSessionInfo = \
                    self.get_model_session_info(grammar_id, model_id)
                last_epoch_completed: int = self.get_last_epoch_completed(
                    grammar_id, model_id)
                trainable_params, non_trainable_params, all_params = \
                    self.get_model_params(grammar_id, model_id)
                for set_name in set_names:
                    labels: List[str] = self.get_labels(grammar_id, model_id,
                                                        set_name)
                    if labels is None:
                        num_labels: int = np.nan
                    else:
                        num_labels: int = len(labels)

                    metrics_loss, metrics_accuracy = self.get_metrics(
                        grammar_id, model_id, set_name)
                    roc_micro_auc, roc_macro_auc = self.get_roc_auc(
                        grammar_id, model_id, set_name)
                    pr_micro_auc, pr_macro_auc = self.get_pr_auc(
                        grammar_id, model_id, set_name)

                    grammar_id_column.append(grammar_id)
                    model_id_column.append(model_id)
                    set_name_column.append(set_name)
                    d_seqgra_version_column.append(
                        data_session_info.seqgra_version)
                    d_numpy_version_column.append(
                        data_session_info.numpy_version)
                    d_python_version_column.append(
                        data_session_info.python_version)
                    d_training_set_size_column.append(training_set_size)
                    d_validation_set_size_column.append(validation_set_size)
                    d_test_set_size_column.append(test_set_size)
                    d_num_labels_column.append(num_labels)
                    m_seqgra_version_column.append(
                        model_session_info.seqgra_version)
                    m_numpy_version_column.append(
                        model_session_info.numpy_version)
                    m_python_version_column.append(
                        model_session_info.python_version)
                    m_library_column.append(model_session_info.library)
                    m_library_version_column.append(
                        model_session_info.library_version)
                    m_last_epoch_completed_column.append(last_epoch_completed)
                    m_trainable_params_column.append(trainable_params)
                    m_non_trainable_params_column.append(non_trainable_params)
                    m_all_params_column.append(all_params)
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
             "d_seqgra_version": d_seqgra_version_column,
             "d_numpy_version": d_numpy_version_column,
             "d_python_version": d_python_version_column,
             "d_training_set_size": d_training_set_size_column,
             "d_validation_set_size": d_validation_set_size_column,
             "d_test_set_size": d_test_set_size_column,
             "d_num_labels": d_num_labels_column,
             "m_seqgra_version": m_seqgra_version_column,
             "m_numpy_version": m_numpy_version_column,
             "m_python_version": m_python_version_column,
             "m_library": m_library_column,
             "m_library_version": m_library_version_column,
             "m_last_epoch_completed": m_last_epoch_completed_column,
             "m_trainable_params": m_trainable_params_column,
             "m_non_trainable_params": m_non_trainable_params_column,
             "m_all_params": m_all_params_column,
             "e_metrics_loss": e_metrics_loss_column,
             "e_metrics_accuracy": e_metrics_accuracy_column,
             "e_roc_micro_auc": e_roc_micro_auc_column,
             "e_roc_macro_auc": e_roc_macro_auc_column,
             "e_pr_micro_auc": e_pr_micro_auc_column,
             "e_pr_macro_auc": e_pr_macro_auc_column})
        df.to_csv(self.output_dir + "table.txt", sep="\t", index=False)

    def get_set_size(self, grammar_id: str, set_name: str) -> int:
        set_file_name: str = self.data_dir + grammar_id + "/" + \
            set_name + ".txt"
        if os.path.isfile(set_file_name):
            # do not count header
            i: int = -1
            with open(set_file_name) as f:
                for line in f:
                    i += 1
            return i
        else:
            self.logger.warning("file does not exist: %s",
                                set_file_name)
            return np.nan

    @staticmethod
    def _read_lines(file_name: str) -> Optional[List[str]]:
        if os.path.isfile(file_name):
            with open(file_name) as f:
                return f.readlines()
        else:
            return None

    @staticmethod
    def _get_property(lines: List[str], name: str) -> str:
        for line in lines:
            if line.startswith(name):
                return line.replace(name, "").strip()

        return ""

    def get_data_session_info(self, grammar_id: str) -> DataSessionInfo:
        file_name: str = self.data_dir + grammar_id + "/session-info.txt"
        lines: List[str] = TableComparator._read_lines(file_name)
        if lines:
            seqgra_version: str = TableComparator._get_property(
                lines, "seqgra package version: ")
            numpy_version: str = TableComparator._get_property(
                lines, "NumPy version: ")
            python_version: str = TableComparator._get_property(
                lines, "Python version: ")
            return DataSessionInfo(seqgra_version, numpy_version,
                                   python_version)
        else:
            self.logger.warning("file does not exist: %s",
                                file_name)
            return DataSessionInfo("", "", "")

    def get_model_session_info(self, grammar_id: str,
                               model_id: str) -> ModelSessionInfo:
        file_name: str = self.model_dir + grammar_id + "/" + \
            model_id + "/session-info.txt"
        lines: List[str] = TableComparator._read_lines(file_name)

        if lines:
            seqgra_version: str = TableComparator._get_property(
                lines, "seqgra package version: ")
            numpy_version: str = TableComparator._get_property(
                lines, "NumPy version: ")
            python_version: str = TableComparator._get_property(
                lines, "Python version: ")
            pytorch_version: str = TableComparator._get_property(
                lines, "PyTorch version: ")
            tf_version: str = TableComparator._get_property(
                lines, "TensorFlow version: ")

            if pytorch_version:
                return ModelSessionInfo(seqgra_version,
                                        numpy_version,
                                        python_version,
                                        "PyTorch",
                                        pytorch_version)
            elif tf_version:
                return ModelSessionInfo(seqgra_version,
                                        numpy_version,
                                        python_version,
                                        "TensorFlow",
                                        tf_version)
            else:
                return ModelSessionInfo(seqgra_version,
                                        numpy_version,
                                        python_version,
                                        "Bayes Optimal Classifier",
                                        seqgra_version)
        else:
            self.logger.warning("file does not exist: %s",
                                file_name)
            return ModelSessionInfo("", "", "", "", "")

    def get_last_epoch_completed(self, grammar_id: str, model_id: str) -> int:
        file_name: str = self.model_dir + grammar_id + "/" + \
            model_id + "/last-epoch-completed.txt"
        last_epoch: int = np.nan
        if os.path.isfile(file_name):
            with open(file_name) as f:
                last_epoch = int(f.readline().strip())
        else:
            self.logger.warning("file does not exist: %s",
                                file_name)

        return last_epoch

    def get_model_params(self, grammar_id: str,
                         model_id: str) -> Tuple[int, int, int]:
        model_params_file_name: str = self.model_dir + grammar_id + "/" + \
            model_id + "/num-model-parameters.txt"
        trainable_params: int = np.nan
        non_trainable_params: int = np.nan
        all_params: int = np.nan
        if os.path.isfile(model_params_file_name):
            with open(model_params_file_name) as f:
                trainable_params = int(f.readline().strip().split("\t")[1])
                non_trainable_params = int(f.readline().strip().split("\t")[1])
                all_params = int(f.readline().strip().split("\t")[1])
        else:
            self.logger.warning("file does not exist: %s",
                                model_params_file_name)

        return trainable_params, non_trainable_params, all_params

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
            self.logger.warning("file does not exist: %s",
                                metrics_file_name)
            return Metrics(np.nan, np.nan)

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
            self.logger.warning("file does not exist: %s",
                                predict_file_name)
            return (np.nan, np.nan)

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
            self.logger.warning("file does not exist: %s",
                                predict_file_name)
            return (np.nan, np.nan)
