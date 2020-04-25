"""Contains abstract classes for all learners.

Classes:
    Learner: abstract base class for all learners
    MultiClassClassificationLearner: abstract class for multi-class
        classification learners
    MultiLabelClassificationLearner: abstract class for multi-label
        classification learners
    MultipleRegressionLearner: abstract class for multiple regression learners
    MultivariateRegressionLearner: abstract class for multivariate
        regression learners
"""
import os
from abc import ABC, abstractmethod
from typing import List, Any, Dict

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp

from seqgra import MiscHelper
from seqgra.model import ModelDefinition
from seqgra.model.model import Architecture


class Learner(ABC):
    """Abstract base class for all learners.

    Attributes:
        id (str): learner ID, used for output folder name
        name (str): learner name
        description (str): concise description of the model architecture
        library (str): TensorFlow or PyTorch
        seed (int): seed for Python, NumPy, and machine learning library
        learner_type (str): one of the following: multi-class classification,
            multi-label classification, multiple regression, multivariate
            regression
        learner_implementation (str): class name of the learner implementation,
            KerasDNAMultiLabelClassificationLearner
        labels (List[str]): class labels expected from output layer
        architecture (Architecture): model architecture
        loss_hyperparameters (Dict[str, str]): hyperparmeters for loss
            function, e.g., type of loss function
        optimizer_hyperparameters (Dict[str, str]): hyperparmeters for
            optimizer, e.g., optimizer type
        training_process_hyperparameters (Dict[str, str]): hyperparmeters
            regarding the training process, e.g., batch size
        data_dir (str): directory with data files, e.g., `training.txt`
        output_dir (str): model output directory,
            `{OUTPUTDIR}/models/{GRAMMAR ID}/{MODEL ID}/`
        model: PyTorch or TensorFlow model
        optimizer: PyTorch or TensorFlow optimizer
        criterion: PyTorch or TensorFlow criterion (loss)
        metrics (List[str]): metrics that are collected, usually `loss` and
            `accuracy`

    Methods:
        train_model
        parse_data
        get_examples_file
        get_annotations_file
        create_model
        save_model
        load_model
        print_model_summary
        predict
        encode_x
        decode_x
        encode_y
        decode_y
        get_num_params
        set_seed

    Arguments:
        parser (ModelParser): parser for model definition
        data_dir (str): directory with data files,
            `{OUTPUTDIR}/input/{GRAMMAR ID}`
        output_dir (str): model output directory without model folder,
            `{OUTPUTDIR}/models/{GRAMMAR ID}`

    See Also:
        :class:`MultiClassClassificationLearner`: for classification models with
            mutually exclusive classes
        MultiLabelClassificationLearner: for classification models with
            non-mutually exclusive classes
        MultipleRegressionLearner: for regression models with multiple
            independent variables and one dependent variable
    """

    @abstractmethod
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        self.definition: ModelDefinition = model_definition
        self.data_dir = MiscHelper.prepare_path(data_dir)
        self.output_dir = MiscHelper.prepare_path(output_dir + "/" +
                                                  self.definition.model_id,
                                                  allow_exists=False)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.metrics = ["loss", "accuracy"]

    def train_model(self,
                    training_file: str = None,
                    validation_file: str = None,
                    x_train: List[str] = None,
                    y_train: List[str] = None,
                    x_val: List[str] = None,
                    y_val: List[str] = None) -> None:
        """Train model.

        Specify either `training_file` and `validation_file` or
        `x_train`, `y_train`, `x_val`, and `y_val`.

        Arguments:
            training_file (str, optional): TODO
            validation_file (str, optional): TODO
            x_train (List[str], optional): TODO
            y_train (List[str], optional): TODO
            x_val (List[str], optional): TODO
            y_val (List[str], optional): TODO

        Raises:
            Exception: output directory non-empty
            Exception: specify either training_file and validation_file
                or x_train, y_train, x_val, y_val
        """
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
        """Abstract method to parse data.

        Sequence data type specific implementations provided for DNA and
        amino acid sequences.

        Arguments:
            file_name (str): file name
        """

    def get_examples_file(self, set_name: str = "test") -> str:
        """Get path to examples file.

        E.g., `get_examples_file("training")` returns
        `{OUTPUTDIR}/input/{GRAMMAR ID}/training.txt`, if it exists.

        Arguments:
            set_name (str, optional): set name can be one of the following:
                `training`, `validation`, or `test`; defaults to `test`

        Returns:
            str: file path to examples file

        Raises:
            Exception: in case requested examples file does not exist
        """
        f: str = self.data_dir + "/" + set_name + ".txt"
        f = f.replace("\\", "/").replace("//", "/").strip()
        if os.path.isfile(f):
            return f
        else:
            raise Exception("examples file does not exist for set " + set_name)

    def get_annotations_file(self, set_name: str = "test") -> str:
        """Get path to annotations file.

        E.g., `get_annotations_file("training")` returns
        `{OUTPUTDIR}/input/{GRAMMAR ID}/training-annotation.txt`, if it exists.

        Arguments:
            set_name (str, optional): set name can be one of the following:
                `training`, `validation`, or `test`; defaults to `test`

        Returns:
            str: file path to annotations file

        Raises:
            Exception: in case requested annotations file does not exist
        """
        f: str = self.data_dir + "/" + set_name + "-annotation.txt"
        f = f.replace("\\", "/").replace("//", "/").strip()
        if os.path.isfile(f):
            return f
        else:
            raise Exception("annotations file does not exist for set " +
                            set_name)

    @abstractmethod
    def create_model(self) -> None:
        """Abstract method to create library-specific model.

        Machine learning library specific implementations are provided for
        TensorFlow and PyTorch.
        """

    @abstractmethod
    def save_model(self, model_name: str = "final"):
        """TODO

        TODO

        Arguments:
            model_name (str, optional): file name in output dir;
                defaults to `final`
        """

    @abstractmethod
    def load_model(self, model_name: str = "final"):
        """TODO

        TODO

        Arguments:
            model_name (str, optional): file name in output dir;
                defaults to `final`
        """

    @abstractmethod
    def print_model_summary(self) -> None:
        """TODO

        TODO
        """

    @abstractmethod
    def predict(self, x: Any, encode: bool = True):
        """TODO

        TODO

        Arguments:
            x (array): TODO
            encode (bool, optional): whether `x` should be encoded;
                defaults to `True`
        """

    @abstractmethod
    def encode_x(self, x):
        """TODO

        TODO

        Arguments:
            x (array): TODO
        """

    @abstractmethod
    def decode_x(self, x):
        """TODO

        TODO

        Arguments:
            x (array): TODO
        """

    @abstractmethod
    def encode_y(self, y):
        """TODO

        TODO

        Arguments:
            y (array): TODO
        """

    @abstractmethod
    def decode_y(self, y):
        """TODO

        TODO

        Arguments:
            y (array): TODO
        """

    @abstractmethod
    def get_num_params(self):
        """TODO

        TODO
        """

    @abstractmethod
    def set_seed(self) -> None:
        """TODO

        TODO
        """

    @abstractmethod
    def _train_model(self,
                     x_train: List[str], y_train: List[str],
                     x_val: List[str], y_val: List[str]) -> None:
        pass


class MultiClassClassificationLearner(Learner):
    """Abstract class for multi-class classification learners.

    Multi-class classification learners are learners for models with
    mututally exclusive class labels.

    Attributes:
        id (str): learner ID, used for output folder name
        name (str): learner name
        description (str): concise description of the model architecture
        library (str): TensorFlow or PyTorch
        seed (int): seed for Python, NumPy, and machine learning library
        learner_type (str): one of the following: multi-class classification,
            multi-label classification, multiple regression, multivariate
            regression
        learner_implementation (str): class name of the learner implementation,
            KerasDNAMultiLabelClassificationLearner
        labels (List[str]): class labels expected from output layer
        architecture (Architecture): model architecture
        loss_hyperparameters (Dict[str, str]): hyperparmeters for loss
            function, e.g., type of loss function
        optimizer_hyperparameters (Dict[str, str]): hyperparmeters for
            optimizer, e.g., optimizer type
        training_process_hyperparameters (Dict[str, str]): hyperparmeters
            regarding the training process, e.g., batch size
        data_dir (str): directory with data files, e.g., `training.txt`
        output_dir (str): model output directory,
            `{OUTPUTDIR}/models/{GRAMMAR ID}/{MODEL ID}/`
        model: PyTorch or TensorFlow model
        optimizer: PyTorch or TensorFlow optimizer
        criterion: PyTorch or TensorFlow criterion (loss)
        metrics (List[str]): metrics that are collected, usually `loss` and
            `accuracy`

    Methods:
        train_model
        parse_data
        get_examples_file
        get_annotations_file
        create_model
        save_model
        load_model
        print_model_summary
        predict
        encode_x
        decode_x
        encode_y
        decode_y
        get_num_params
        set_seed
        evaluate_model
        create_roc_curve
        create_precision_recall_curve

    Arguments:
        parser (ModelParser): parser for model definition
        data_dir (str): directory with data files,
            `{OUTPUTDIR}/input/{GRAMMAR ID}`
        output_dir (str): model output directory without model folder,
            `{OUTPUTDIR}/models/{GRAMMAR ID}`
    """

    @abstractmethod
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

        if self.definition.learner_type != "multi-class classification":
            raise Exception("model definition must specify multi-class "
                            "classification learner type, but learner type "
                            "is '" + self.definition.learner_type, "'")

    def evaluate_model(self, file_name: str = None,
                       x: List[str] = None, y: List[str] = None):
        """TODO

        TODO

        Arguments:
            file_name (str): TODO
            x (List[str]): TODO
            y (List[str]): TODO

        Returns:
            array: TODO

        Raises:
            Exception: if neither `file_name` nor (`x` and `y`) are specified
        """
        if file_name is not None:
            x, y = self.parse_data(file_name)

        if x is None or y is None:
            raise Exception(
                "specify either file_name or x and y")
        else:
            return self._evaluate_model(x, y)

    def create_roc_curve(self, y_true, y_hat, file_name: str) -> None:
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
        n_classes = len(self.definition.labels)
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

        line, _ = plt.plot(fpr["micro"], tpr["micro"],
                           color="gold", linestyle=":", linewidth=2)
        lines.append(line)
        labels.append("micro-average (area = {0:0.2f})"
                      "".format(roc_auc["micro"]))

        line, _ = plt.plot(fpr["macro"], tpr["macro"],
                           color="darkorange", linestyle=":", linewidth=2)
        lines.append(line)
        labels.append("macro-average (area = {0:0.2f})"
                      "".format(roc_auc["macro"]))

        for i in range(n_classes):
            line, _ = plt.plot(fpr[i], tpr[i], linewidth=2)
            lines.append(line)
            labels.append("condition {0} (area = {1:0.2f})"
                          "".format(self.definition.labels[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], "k--", linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(lines, labels, bbox_to_anchor=(1.04, 1),
                   loc="upper left", prop=dict(size=14))
        plt.savefig(file_name, bbox_inches="tight")

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
        n_classes = len(self.definition.labels)
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
            line, _ = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate(r"$F_1 = {0:0.1f}$".format(
                f_score), xy=(0.89, y[45] + 0.02))

        lines.append(line)
        labels.append(r"iso-$F_1$ curves")
        line, _ = plt.plot(recall["micro"], precision["micro"],
                           linestyle=":", color="gold", linewidth=2)
        lines.append(line)
        labels.append("micro-average (area = {0:0.2f})"
                      "".format(average_precision["micro"]))

        for i in range(n_classes):
            line, _ = plt.plot(recall[i], precision[i], linewidth=2)
            lines.append(line)
            labels.append("condition {0} (area = {1:0.2f})"
                          "".format(self.definition.labels[i], average_precision[i]))

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
    """Abstract class for multi-label classification learners.

    Multi-label classification learners are learners for models with class
    labels that are not mututally exclusive.

    Attributes:
        id (str): learner ID, used for output folder name
        name (str): learner name
        description (str): concise description of the model architecture
        library (str): TensorFlow or PyTorch
        seed (int): seed for Python, NumPy, and machine learning library
        learner_type (str): one of the following: multi-class classification,
            multi-label classification, multiple regression, multivariate
            regression
        learner_implementation (str): class name of the learner implementation,
            KerasDNAMultiLabelClassificationLearner
        labels (List[str]): class labels expected from output layer
        architecture (Architecture): model architecture
        loss_hyperparameters (Dict[str, str]): hyperparmeters for loss
            function, e.g., type of loss function
        optimizer_hyperparameters (Dict[str, str]): hyperparmeters for
            optimizer, e.g., optimizer type
        training_process_hyperparameters (Dict[str, str]): hyperparmeters
            regarding the training process, e.g., batch size
        data_dir (str): directory with data files, e.g., `training.txt`
        output_dir (str): model output directory,
            `{OUTPUTDIR}/models/{GRAMMAR ID}/{MODEL ID}/`
        model: PyTorch or TensorFlow model
        optimizer: PyTorch or TensorFlow optimizer
        criterion: PyTorch or TensorFlow criterion (loss)
        metrics (List[str]): metrics that are collected, usually `loss` and
            `accuracy`

    Methods:
        train_model
        parse_data
        get_examples_file
        get_annotations_file
        create_model
        save_model
        load_model
        print_model_summary
        predict
        encode_x
        decode_x
        encode_y
        decode_y
        get_num_params
        set_seed
        evaluate_model
        create_roc_curve
        create_precision_recall_curve

    Arguments:
        parser (ModelParser): parser for model definition
        data_dir (str): directory with data files,
            `{OUTPUTDIR}/input/{GRAMMAR ID}`
        output_dir (str): model output directory without model folder,
            `{OUTPUTDIR}/models/{GRAMMAR ID}`
    """

    @abstractmethod
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

        if self.definition.learner_type != "multi-label classification":
            raise Exception("model definition must specify multi-label "
                            "classification learner type, but learner type "
                            "is '" + self.definition.learner_type, "'")

    def evaluate_model(self, file_name: str = None,
                       x: List[str] = None, y: List[List[str]] = None):
        """TODO

        TODO

        Arguments:
            file_name (str): TODO
            x (List[str]): TODO
            y (List[List[str]]): TODO

        Returns:
            array: TODO

        Raises:
            Exception: if neither `file_name` nor (`x` and `y`) are specified
        """
        if file_name is not None:
            x, y = self.parse_data(file_name)

        if x is None or y is None:
            raise Exception(
                "specify either file_name or x and y")
        else:
            return self._evaluate_model(x, y)

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
        n_classes = len(self.definition.labels)
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

        line, _ = plt.plot(fpr["micro"], tpr["micro"],
                           color="gold", linestyle=":", linewidth=2)
        lines.append(line)
        labels.append("micro-average (area = {0:0.2f})"
                      "".format(roc_auc["micro"]))

        line, _ = plt.plot(fpr["macro"], tpr["macro"],
                           color="darkorange", linestyle=":", linewidth=2)
        lines.append(line)
        labels.append("macro-average (area = {0:0.2f})"
                      "".format(roc_auc["macro"]))

        for i in range(n_classes):
            line, _ = plt.plot(fpr[i], tpr[i], linewidth=2)
            lines.append(line)
            labels.append("condition {0} (area = {1:0.2f})"
                          "".format(self.definition.labels[i], roc_auc[i]))

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
        n_classes = len(self.definition.labels)
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
            line, _ = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
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
                          "".format(self.definition.labels[i], average_precision[i]))

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
    def _evaluate_model(self, x: List[str], y: List[List[str]]):
        pass


class MultipleRegressionLearner(Learner):
    """Abstract class for multiple regression learners.

    Multiple regression learners are learners for models with
    multiple independent real-valued variables (:math:`x \\in R^n`) and
    one dependent real-valued variable (:math:`x \\in R`).

    Attributes:
        id (str): learner ID, used for output folder name
        name (str): learner name
        description (str): concise description of the model architecture
        library (str): TensorFlow or PyTorch
        seed (int): seed for Python, NumPy, and machine learning library
        learner_type (str): one of the following: multi-class classification,
            multi-label classification, multiple regression, multivariate
            regression
        learner_implementation (str): class name of the learner implementation,
            KerasDNAMultiLabelClassificationLearner
        labels (List[str]): class labels expected from output layer
        architecture (Architecture): model architecture
        loss_hyperparameters (Dict[str, str]): hyperparmeters for loss
            function, e.g., type of loss function
        optimizer_hyperparameters (Dict[str, str]): hyperparmeters for
            optimizer, e.g., optimizer type
        training_process_hyperparameters (Dict[str, str]): hyperparmeters
            regarding the training process, e.g., batch size
        data_dir (str): directory with data files, e.g., `training.txt`
        output_dir (str): model output directory,
            `{OUTPUTDIR}/models/{GRAMMAR ID}/{MODEL ID}/`
        model: PyTorch or TensorFlow model
        optimizer: PyTorch or TensorFlow optimizer
        criterion: PyTorch or TensorFlow criterion (loss)
        metrics (List[str]): metrics that are collected, usually `loss` and
            `accuracy`

    Methods:
        train_model
        parse_data
        get_examples_file
        get_annotations_file
        create_model
        save_model
        load_model
        print_model_summary
        predict
        encode_x
        decode_x
        encode_y
        decode_y
        get_num_params
        set_seed
        evaluate_model
        create_roc_curve
        create_precision_recall_curve

    Arguments:
        parser (ModelParser): parser for model definition
        data_dir (str): directory with data files,
            `{OUTPUTDIR}/input/{GRAMMAR ID}`
        output_dir (str): model output directory without model folder,
            `{OUTPUTDIR}/models/{GRAMMAR ID}`
    """

    @abstractmethod
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

        if self.definition.learner_type != "multiple regression":
            raise Exception("model definition must specify multiple "
                            "regression learner type, but learner type "
                            "is '" + self.definition.learner_type, "'")

    def evaluate_model(self, file_name: str = None,
                       x: List[str] = None, y: List[float] = None):
        """TODO

        TODO

        Arguments:
            file_name (str): TODO
            x (List[str]): TODO
            y (List[float]): TODO

        Returns:
            array: TODO

        Raises:
            Exception: if neither `file_name` nor (`x` and `y`) are specified
        """
        if file_name is not None:
            x, y = self.parse_data(file_name)

        if x is None or y is None:
            raise Exception("specify either file_name or x and y")
        else:
            return self._evaluate_model(x, y)

    def create_roc_curve(self, y_true, y_hat, file_name) -> None:
        """Create ROC curve.

        Plots ROC curves for each class label, including micro-average and
        macro-average. Saves plot as PDF in `file_name`.

        Arguments:
            y_true (array): TODO ; shape = [n_samples, n_classes]
            y_hat (array): TODO ; shape = [n_samples, n_classes]
            file_name (str): TODO
        """

    def create_precision_recall_curve(self, y_true, y_hat, file_name) -> None:
        """Create precision-recall curve.

        Plots PR curves for each class label, including micro-average and
        iso-F1 curves. Saves plot as PDF in `file_name`.

        Arguments:
            y_true (array): TODO ; shape = [n_samples, n_classes]
            y_hat (array): TODO ; shape = [n_samples, n_classes]
            file_name (str): TODO
        """

    @abstractmethod
    def _evaluate_model(self, x: List[str], y: List[float]):
        pass


class MultivariateRegressionLearner(Learner):
    """Abstract class for multivariate regression learners.

    Multivariate regression learners are used for models with
    multiple independent real-valued variables (:math:`x \\in R^n`) and
    multiple dependent real-valued variables (:math:`y \\in R^n`).

    Attributes:
        id (str): learner ID, used for output folder name
        name (str): learner name
        description (str): concise description of the model architecture
        library (str): TensorFlow or PyTorch
        seed (int): seed for Python, NumPy, and machine learning library
        learner_type (str): one of the following: multi-class classification,
            multi-label classification, multiple regression, multivariate
            regression
        learner_implementation (str): class name of the learner implementation,
            KerasDNAMultiLabelClassificationLearner
        labels (List[str]): class labels expected from output layer
        architecture (Architecture): model architecture
        loss_hyperparameters (Dict[str, str]): hyperparmeters for loss
            function, e.g., type of loss function
        optimizer_hyperparameters (Dict[str, str]): hyperparmeters for
            optimizer, e.g., optimizer type
        training_process_hyperparameters (Dict[str, str]): hyperparmeters
            regarding the training process, e.g., batch size
        data_dir (str): directory with data files, e.g., `training.txt`
        output_dir (str): model output directory,
            `{OUTPUTDIR}/models/{GRAMMAR ID}/{MODEL ID}/`
        model: PyTorch or TensorFlow model
        optimizer: PyTorch or TensorFlow optimizer
        criterion: PyTorch or TensorFlow criterion (loss)
        metrics (List[str]): metrics that are collected, usually `loss` and
            `accuracy`

    Methods:
        train_model
        parse_data
        get_examples_file
        get_annotations_file
        create_model
        save_model
        load_model
        print_model_summary
        predict
        encode_x
        decode_x
        encode_y
        decode_y
        get_num_params
        set_seed
        evaluate_model
        create_roc_curve
        create_precision_recall_curve

    Arguments:
        parser (ModelParser): parser for model definition
        data_dir (str): directory with data files,
            `{OUTPUTDIR}/input/{GRAMMAR ID}`
        output_dir (str): model output directory without model folder,
            `{OUTPUTDIR}/models/{GRAMMAR ID}`
    """

    @abstractmethod
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

        if self.definition.learner_type != "multivariate regression":
            raise Exception("model definition must specify multivariate "
                            "regression learner type, but learner type "
                            "is '" + self.definition.learner_type, "'")

    def evaluate_model(self, file_name: str = None,
                       x: List[str] = None, y: List[List[float]] = None):
        """TODO

        TODO

        Arguments:
            file_name (str): TODO
            x (List[str]): TODO
            y (List[List[float]]): TODO

        Returns:
            array: TODO

        Raises:
            Exception: if neither `file_name` nor (`x` and `y`) are specified
        """
        if file_name is not None:
            x, y = self.parse_data(file_name)

        if x is None or y is None:
            raise Exception(
                "specify either file_name or x and y")
        else:
            return self._evaluate_model(x, y)

    def create_roc_curve(self, y_true, y_hat, file_name) -> None:
        """Create ROC curve.

        Plots ROC curves for each class label, including micro-average and
        macro-average. Saves plot as PDF in `file_name`.

        Arguments:
            y_true (array): TODO ; shape = [n_samples, n_classes]
            y_hat (array): TODO ; shape = [n_samples, n_classes]
            file_name (str): TODO
        """

    def create_precision_recall_curve(self, y_true, y_hat, file_name) -> None:
        """Create precision-recall curve.

        Plots PR curves for each class label, including micro-average and
        iso-F1 curves. Saves plot as PDF in `file_name`.

        Arguments:
            y_true (array): TODO ; shape = [n_samples, n_classes]
            y_hat (array): TODO ; shape = [n_samples, n_classes]
            file_name (str): TODO
        """

    @abstractmethod
    def _evaluate_model(self, x: List[str], y: List[float]):
        pass
