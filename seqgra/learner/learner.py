"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all learners
- abstract class for multi-class classification learners, i.e., learners
  for data where the class labels are mututally exclusive
- abstract class for multi-label classification learners, i.e., learners
  for data where the class labels are not mututally exclusive
- abstract class for multiple regression learners, i.e., learners with
  multiple independent variables and one dependent variable
- abstract class for multivariate regression learners, i.e., learners with
  multiple independent variables and multiple dependent variables

@author: Konstantin Krismer
"""
import os
from abc import ABC, abstractmethod
from typing import List, Any, Dict

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from scipy import interp

from seqgra.parser.modelparser import ModelParser
from seqgra.model.model.architecture import Architecture
from seqgra.model.model.operation import Operation


class Learner(ABC):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        self._parser: ModelParser = parser
        self.__parse_config()
        output_dir = output_dir.strip().replace("\\", "/")
        if not output_dir.endswith("/"):
            output_dir += "/"
        self.output_dir = output_dir + self.id + "/"
        self.__prepare_output_dir()

    def __str__(self):
        str_rep = ["seqgra model configuration:\n",
                   "\tID: ", self.id, "\n",
                   "\tLabel: ", self.label, "\n",
                   "\tDescription:\n"]
        if self.description:
            str_rep += ["\t", self.description, "\n"]
        str_rep += ["\tLibrary: ", self.library, "\n",
                    "\tLearner type: ", self.learner_type, "\t",
                    "\tLearner implementation", self.learner_implementation]

        str_rep += ["\tMetrics:\n", str(self.metrics)]

        str_rep += ["\t" + s +
                    "\n" for s in str(self.architecture).splitlines()]

        str_rep += ["\tLoss hyperparameters:\n", "\t\t",
                    str(self.loss_hyperparameters), "\n"]
        str_rep += ["\tOptimizer hyperparameters:\n", "\t\t",
                    str(self.optimizer_hyperparameters), "\n"]
        str_rep += ["\tTraining process hyperparameters:\n", "\t\t",
                    str(self.training_process_hyperparameters), "\n"]

        return "".join(str_rep)

    def train_model(self,
                    training_file: str = None,
                    validation_file: str = None,
                    x_train: List[str] = None,
                    y_train: List[str] = None,
                    x_val: List[str] = None,
                    y_val: List[str] = None) -> None:
        if len(os.listdir(self.output_dir)) > 0:
            raise Exception("output directory non-empty")

        self.set_seed()

        if training_file is not None:
            x_train, y_train = self.parse_data(training_file)

        if validation_file is not None:
            x_val, y_val = self.parse_data(validation_file)

        if x_train is None or y_train is None or \
           x_val is None or y_val is None:
            raise Exception("specify either training_file and validation_file"
                            " or x_train, y_train, x_val, y_val")
        else:
            self._train_model(x_train, y_train, x_val, y_val)

    @abstractmethod
    def parse_data(self, file_name: str) -> None:
        pass

    @abstractmethod
    def create_model(self) -> None:
        pass

    @abstractmethod
    def save_model(self, model_name: str = "final"):
        pass

    @abstractmethod
    def load_model(self, model_name: str = "final"):
        pass

    @abstractmethod
    def print_model_summary(self) -> None:
        pass

    @abstractmethod
    def predict(self, x: Any, encode: bool = True):
        pass

    @abstractmethod
    def encode_x(self, x):
        pass

    @abstractmethod
    def decode_x(self, x):
        pass
    
    @abstractmethod
    def encode_y(self, y):
        pass
        
    @abstractmethod
    def decode_y(self, y):
        pass

    @abstractmethod
    def get_num_params(self):
        pass

    @abstractmethod
    def set_seed(self) -> None:
        pass

    @abstractmethod
    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        pass

    def __parse_config(self):
        self.id: str = self._parser.get_id()
        self.label: str = self._parser.get_label()
        self.description: str = self._parser.get_description()
        self.library: str = self._parser.get_library()
        self.seed: int = self._parser.get_seed()
        self.learner_type: str = self._parser.get_learner_type()
        self.learner_implementation: str = \
            self._parser.get_learner_implementation()
        self.labels: List[str] = self._parser.get_labels()
        self.metrics: List[str] = self._parser.get_metrics()
        self.architecture: Architecture = self._parser.get_architecture()
        self.loss_hyperparameters: Dict[str, str] = \
            self._parser.get_loss_hyperparameters()
        self.optimizer_hyperparameters: Dict[str, str] = \
            self._parser.get_optimizer_hyperparameters()
        self.training_process_hyperparameters: Dict[str, str] = \
            self._parser.get_training_process_hyperparameters()

    def __prepare_output_dir(self) -> None:
        if os.path.exists(self.output_dir):
            if not os.path.isdir(self.output_dir):
                raise Exception("output directory cannot be created "
                                "(file with same name exists)")
        else:
            os.makedirs(self.output_dir)


class MultiClassClassificationLearner(Learner):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

        if self.learner_type != "multi-class classification":
            raise Exception("model definition must specify multi-class "
                            "classification learner type, but learner type "
                            "is '" + self.learner_type, "'")

    def evaluate_model(self, file_name: str = None,
                       x: List[str] = None, y: List[str] = None):
        if file_name is not None:
            x, y = self.parse_data(file_name)

        if x is None or y is None:
            raise Exception(
                "specify either file_name or x and y")
        else:
            return self._evaluate_model(x, y)

    def create_roc_curve(self, y_true, y_hat, file_name) -> None:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(self.labels)
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
        plt.figure()
        lines = []
        labels = []

        l, = plt.plot(fpr["micro"], tpr["micro"],
                      color="gold", linestyle=":", linewidth=2)
        lines.append(l)
        labels.append("micro-average (area = {0:0.2f})"
                      "".format(roc_auc["micro"]))

        l, = plt.plot(fpr["macro"], tpr["macro"],
                      color="darkorange", linestyle=":", linewidth=2)
        lines.append(l)
        labels.append("macro-average (area = {0:0.2f})"
                      "".format(roc_auc["macro"]))

        for i in range(n_classes):
            l, = plt.plot(fpr[i], tpr[i], linewidth=2)
            lines.append(l)
            labels.append("condition {0} (area = {1:0.2f})"
                          "".format(self.labels[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], "k--", linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(lines, labels, bbox_to_anchor=(1.04, 1),
                   loc="upper left", prop=dict(size=14))
        plt.savefig(file_name, bbox_inches="tight")

    def create_precision_recall_curve(self, y_true, y_hat, file_name) -> None:
        precision = dict()
        recall = dict()
        average_precision = dict()
        n_classes = len(self.labels)
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

        plt.figure(figsize=(7, 8))
        lines = []
        labels = []

        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.001, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate(r"$F_1 = {0:0.1f}$".format(
                f_score), xy=(0.89, y[45] + 0.02))

        lines.append(l)
        labels.append(r"iso-$F_1$ curves")
        l, = plt.plot(recall["micro"], precision["micro"],
                      linestyle=":", color="gold", linewidth=2)
        lines.append(l)
        labels.append("micro-average (area = {0:0.2f})"
                      "".format(average_precision["micro"]))

        for i in range(n_classes):
            l, = plt.plot(recall[i], precision[i], linewidth=2)
            lines.append(l)
            labels.append("condition {0} (area = {1:0.2f})"
                          "".format(self.labels[i], average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend(lines, labels, bbox_to_anchor=(1.04, 1),
                   loc="upper left", prop=dict(size=14))
        plt.savefig(file_name, bbox_inches="tight")

    @abstractmethod
    def _evaluate_model(self, x: List[str], y: List[str]):
        pass


class MultiLabelClassificationLearner(Learner):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

        if self.learner_type != "multi-label classification":
            raise Exception("model definition must specify multi-label "
                            "classification learner type, but learner type "
                            "is '" + self.learner_type, "'")

    def evaluate_model(self, file_name: str = None,
                       x: List[str] = None, y: List[List[str]] = None):
        if file_name is not None:
            x, y = self.parse_data(file_name)

        if x is None or y is None:
            raise Exception(
                "specify either file_name or x and y")
        else:
            return self._evaluate_model(x, y)

    def create_roc_curve(self, y_true, y_hat, file_name) -> None:
        pass

    def create_precision_recall_curve(self, y_true, y_hat, file_name) -> None:
        pass

    @abstractmethod
    def _evaluate_model(self, x: List[str], y: List[List[str]]):
        pass


class MultipleRegressionLearner(Learner):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

        if self.learner_type != "multiple regression":
            raise Exception("model definition must specify multiple "
                            "regression learner type, but learner type "
                            "is '" + self.learner_type, "'")

    def evaluate_model(self, file_name: str = None,
                       x: List[str] = None, y: List[float] = None):
        if file_name is not None:
            x, y = self.parse_data(file_name)

        if x is None or y is None:
            raise Exception(
                "specify either file_name or x and y")
        else:
            return self._evaluate_model(x, y)

    def create_roc_curve(self, y_true, y_hat, file_name) -> None:
        pass

    def create_precision_recall_curve(self, y_true, y_hat, file_name) -> None:
        pass

    @abstractmethod
    def _evaluate_model(self, x: List[str], y: List[float]):
        pass


class MultivariateRegressionLearner(Learner):
    @abstractmethod
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

        if self.learner_type != "multivariate regression":
            raise Exception("model definition must specify multivariate "
                            "regression learner type, but learner type "
                            "is '" + self.learner_type, "'")

    def evaluate_model(self, file_name: str = None,
                       x: List[str] = None, y: List[List[float]] = None):
        if file_name is not None:
            x, y = self.parse_data(file_name)

        if x is None or y is None:
            raise Exception(
                "specify either file_name or x and y")
        else:
            return self._evaluate_model(x, y)

    def create_roc_curve(self, y_true, y_hat, file_name) -> None:
        pass

    def create_precision_recall_curve(self, y_true, y_hat, file_name) -> None:
        pass

    @abstractmethod
    def _evaluate_model(self, x: List[str], y: List[float]):
        pass
