"""
MIT - CSAIL - Gifford Lab - seqgra

- class for all saliency evaluators
- models must be in pytorch

@author: Jennifer Hammelman
"""
from typing import Any, List

import math
import numpy as np
import os
import pandas as pd
import pkg_resources
import subprocess
import torch

import seqgra.constants as c
from seqgra.learner import Learner
from seqgra.evaluator import FeatureImportanceEvaluator
from seqgra.evaluator.explainer.backprop import VanillaGradExplainer
from seqgra.evaluator.explainer.backprop import GradxInputExplainer
from seqgra.evaluator.explainer.backprop import SaliencyExplainer
from seqgra.evaluator.explainer.backprop import IntegrateGradExplainer
from seqgra.evaluator.explainer.backprop import NonlinearIntegrateGradExplainer
from seqgra.evaluator.explainer.deeplift import DeepLIFTRescaleExplainer
from seqgra.evaluator.explainer.gradcam import GradCAMExplainer
from seqgra.evaluator.explainer.ebp import ExcitationBackpropExplainer
from seqgra.evaluator.explainer.ebp import ContrastiveExcitationBackpropExplainer


class GradientBasedEvaluator(FeatureImportanceEvaluator):
    def __init__(self, evaluator_id: str, evaluator_name: str,
                 learner: Learner, output_dir: str) -> None:
        super().__init__(evaluator_id, evaluator_name, learner, output_dir,
                         supported_libraries=[c.LibraryType.TORCH])
        self.explainer = None
        self.relevance_threshold = 0.1

    def _evaluate_model(self, x: List[str], y: List[str],
                        annotations: List[str]) -> Any:
        use_cuda = torch.cuda.is_available()

        # encode
        encoded_x = self.learner.encode_x(x)
        encoded_y = self.learner.encode_y(y)

        # convert bool to float32 and long, as expected by explainers
        encoded_x = encoded_x.astype(np.float32)
        encoded_y = encoded_y.astype(np.int64)

        self._check_tensor_dimensions(encoded_x)
        # convert np array to torch tensor
        encoded_x = torch.from_numpy(encoded_x)
        encoded_y = torch.from_numpy(encoded_y)

        if use_cuda:
            # store input tensor, label tensor and model on GPU
            encoded_x = encoded_x.cuda()
            encoded_y = encoded_y.cuda()
            self.explainer.model.cuda()

        encoded_x = torch.autograd.Variable(encoded_x, requires_grad=True)

        # enable inference mode
        self.explainer.model.eval()

        fi_matrix = self.calculate_saliency(encoded_x, encoded_y)

        self._check_tensor_dimensions(fi_matrix)
        fi_matrix = self._convert_to_nwc(fi_matrix)

        return (fi_matrix, x, y, annotations)

    def _convert_to_nwc(self, x) -> Any:
        if self.learner.definition.library == c.LibraryType.TENSORFLOW and \
                self.learner.definition.input_encoding == "2D":
            # from (N, H, W, C) to (N, W, C)
            x = np.squeeze(x, axis=1)
        elif self.learner.definition.library == c.LibraryType.TORCH:
            if self.learner.definition.input_encoding == "2D":
                # from (N, C, 1, W) to (N, C, W)
                x = np.squeeze(x, axis=2)

            # from (N, C, W) to (N, W, C)
            x = np.transpose(x, (0, 2, 1))

        return x

    @staticmethod
    def _calculate_smooth_precision(normalized_fi_vector,
                                    annotation: str) -> float:
        grammar_positions: List[int] = [i
                                        for i, char in enumerate(annotation)
                                        if char == c.PositionType.GRAMMAR]
        non_grammar_positions: List[int] = [i
                                            for i, char in enumerate(annotation)
                                            if char != c.PositionType.GRAMMAR]
        total_fi: float = normalized_fi_vector.sum()
        grammar_fi: float = normalized_fi_vector[grammar_positions].sum()
        non_grammar_fi: float = normalized_fi_vector[non_grammar_positions].sum(
        )

        if not grammar_positions and not math.isclose(non_grammar_fi, 0.0):
            return 1.0
        elif not grammar_positions:
            return 0.0
        else:
            return grammar_fi / total_fi

    @staticmethod
    def _calculate_smooth_recall(normalized_fi_vector,
                                 annotation: str) -> float:
        grammar_positions: List[int] = [i
                                        for i, char in enumerate(annotation)
                                        if char == c.PositionType.GRAMMAR]
        non_grammar_positions: List[int] = [i
                                            for i, char in enumerate(annotation)
                                            if char != c.PositionType.GRAMMAR]
        grammar_fi: float = normalized_fi_vector[grammar_positions].sum()
        non_grammar_fi: float = normalized_fi_vector[non_grammar_positions].sum(
        )

        if not grammar_positions and math.isclose(non_grammar_fi, 0.0):
            return 1.0
        elif not grammar_positions:
            return 0.0
        else:
            return grammar_fi / float(len(grammar_positions))

    @staticmethod
    def _calculate_smooth_f1(normalized_fi_vector,
                             annotation: str) -> float:
        precision: float = GradientBasedEvaluator._calculate_smooth_precision(
            normalized_fi_vector, annotation)
        recall: float = GradientBasedEvaluator._calculate_smooth_recall(
            normalized_fi_vector, annotation)

        if not precision and not recall:
            return 0.0
        else:
            return 2.0 * ((precision * recall) / (precision + recall))

    @staticmethod
    def _calculate_smooth_specificity(normalized_fi_vector,
                                      annotation: str) -> float:
        grammar_positions: List[int] = [i
                                        for i, char in enumerate(annotation)
                                        if char == c.PositionType.GRAMMAR]
        non_grammar_positions: List[int] = [i
                                            for i, char in enumerate(annotation)
                                            if char != c.PositionType.GRAMMAR]
        grammar_fi: float = normalized_fi_vector[grammar_positions].sum()
        non_grammar_fi: float = normalized_fi_vector[non_grammar_positions].sum(
        )
        true_negative: float = float(
            len(non_grammar_positions)) - non_grammar_fi
        false_negative: float = float(len(grammar_positions)) - grammar_fi

        if math.isclose(true_negative, 0.0) and \
            math.isclose(non_grammar_fi, 0.0) and \
                math.isclose(false_negative, 0.0):
            return 1.0
        if math.isclose(true_negative, 0.0) and \
                math.isclose(non_grammar_fi, 0.0):
            return 0.0
        else:
            return true_negative / float(len(non_grammar_positions))

    @staticmethod
    def _prepare_normalized_fi_vector(fi_matrix) -> Any:
        fi_matrix = np.clip(fi_matrix, a_min=0.0, a_max=None)
        fi_vector = fi_matrix.sum(axis=1)
        norm: float = fi_vector.max()
        fi_vector = fi_vector * (1 / norm)
        return fi_vector

    def _write_result_df(self, fi_matrix, x: List[str], y: List[str],
                         annotations: List[str], set_name: str = "test") -> None:

        precision_column: List[float] = list()
        recall_column: List[float] = list()
        specificity_column: List[float] = list()
        f1_column: List[float] = list()

        for example_id, annotation in enumerate(annotations):
            fi_vector = GradientBasedEvaluator._prepare_normalized_fi_vector(
                fi_matrix[example_id, :, :])
            precision_column += [GradientBasedEvaluator._calculate_smooth_precision(
                fi_vector, annotation)]
            recall_column += [GradientBasedEvaluator._calculate_smooth_recall(
                fi_vector, annotation)]
            specificity_column += [GradientBasedEvaluator._calculate_smooth_specificity(
                fi_vector, annotation)]
            f1_column += [GradientBasedEvaluator._calculate_smooth_f1(
                fi_vector, annotation)]

        df = pd.DataFrame({"x": x,
                           "y": y,
                           "annotation": annotations,
                           "precision": precision_column,
                           "recall": recall_column,
                           "specificity": specificity_column,
                           "f1": f1_column})

        df.to_csv(self.output_dir + set_name + "-df.txt", sep="\t",
                  index=False)

    def _save_results(self, results, set_name: str = "test",
                      suppress_plots: bool = False) -> None:
        np.save(self.output_dir + set_name + "-feature-importance-matrix.npy",
                results[0])
        self._write_result_df(results[0], results[1], results[2], results[3],
                              set_name)

    def _visualize_grammar_agreement(self, results,
                                     set_name: str = "test") -> None:
        super()._visualize_grammar_agreement(results, set_name)

        df: pd.DataFrame = self._convert_to_unthresholded_data_frame(results)
        if len(df.index) > 0:
            df.to_csv(self.output_dir + set_name +
                      "-grammar-agreement-df.txt",
                      sep="\t", index=False)

            if self.is_ggplot_available:
                plot_script: str = pkg_resources.resource_filename(
                    "seqgra", "evaluator/plotagreement.R")
                temp_file_name: str = self.output_dir + set_name + "-temp.txt"
                pdf_file_name: str = self.output_dir + set_name + \
                    "-grammar-agreement.pdf"
                df: pd.DataFrame = self._prepare_unthresholded_r_data_frame(df)
                df.to_csv(temp_file_name, sep="\t", index=False)
                cmd = ["Rscript", "--vanilla", plot_script, temp_file_name,
                       pdf_file_name, self.evaluator_name]
                subprocess.call(cmd, universal_newlines=True)
                os.remove(temp_file_name)

    def _prepare_unthresholded_r_data_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df["precision"] = 0.0
        df["recall"] = 0.0
        df["specificity"] = 0.0
        df["f1"] = 0.0
        df["n"] = 0

        for example_id in set(df.example.tolist()):
            example_df = df.loc[df.example == example_id]
            fi_vector = np.asarray(example_df.value.tolist(), dtype=np.float32)
            annotation: str = "".join(example_df.group.tolist())
            df.loc[df.example == example_id, "precision"] = \
                GradientBasedEvaluator._calculate_smooth_precision(
                    fi_vector, annotation)
            df.loc[df.example == example_id, "recall"] = \
                GradientBasedEvaluator._calculate_smooth_recall(
                    fi_vector, annotation)
            df.loc[df.example == example_id, "specificity"] = \
                GradientBasedEvaluator._calculate_smooth_specificity(
                    fi_vector, annotation)
            df.loc[df.example == example_id, "f1"] = \
                GradientBasedEvaluator._calculate_smooth_f1(
                    fi_vector, annotation)
            df.loc[df.example == example_id, "n"] = 1.0 / len(example_df.index)

        df["precision"] = df.groupby("label")["precision"].transform("mean")
        df["recall"] = df.groupby("label")["recall"].transform("mean")
        df["specificity"] = df.groupby(
            "label")["specificity"].transform("mean")
        df["f1"] = df.groupby("label")["f1"].transform("mean")
        df["n"] = round(df.groupby("label")["n"].transform("sum"))

        return df

    def calculate_saliency(self, data, label):
        result = self.explainer.explain(data, label)
        return self._explainer_transform(data, result)

    def _explainer_transform(self, data, result):
        return result.cpu().numpy()

    def __get_agreement_group(self, annotation_position: str,
                              importance_vector) -> str:
        if annotation_position == c.PositionType.GRAMMAR:
            if np.max(importance_vector) < self.relevance_threshold:
                return "FN"
            else:
                return "TP"
        else:
            if np.max(importance_vector) < self.relevance_threshold:
                return "TN"
            else:
                return "FP"

    def _check_tensor_dimensions(self, tensor) -> None:
        if self.learner.definition.library == c.LibraryType.TENSORFLOW:
            if self.learner.definition.input_encoding == "2D":
                expected_shape: str = "(N, 1, W, C)"
                channel_dim: int = 3
                height_dim: int = 1
                n_dims: int = 4
            else:
                expected_shape: str = "(N, W, C)"
                channel_dim: int = 2
                height_dim: int = None
                n_dims: int = 3
        elif self.learner.definition.library == c.LibraryType.TORCH:
            channel_dim: int = 1
            if self.learner.definition.input_encoding == "2D":
                expected_shape: str = "(N, C, 1, W)"
                height_dim: int = 2
                n_dims: int = 4
            else:
                expected_shape: str = "(N, C, W)"
                height_dim: int = None
                n_dims: int = 3

        if len(tensor.shape) != n_dims:
            raise Exception("tensor shape invalid: expected " +
                            expected_shape + ", got " +
                            str(tensor.shape))

        if height_dim and tensor.shape[height_dim] != 1:
            raise Exception("tensor shape invalid: expected "
                            "height dimension size of 1, got " +
                            str(tensor.shape[height_dim]))

        if self.learner.definition.sequence_space == c.SequenceSpaceType.DNA:
            if tensor.shape[channel_dim] != 4:
                raise Exception("tensor shape invalid for DNA "
                                "sequence space: expected 4 channels, got " +
                                str(tensor.shape[channel_dim]))
        elif self.learner.definition.sequence_space == c.SequenceSpaceType.PROTEIN:
            if tensor.shape[channel_dim] != 20:
                raise Exception("tensor shape invalid for protein "
                                "sequence space: expected 20 "
                                "channels, got " +
                                str(tensor.shape[channel_dim]))

    def _convert_to_data_frame(self, results) -> pd.DataFrame:
        """Takes gradient-based evaluator-specific results and turns them into
        a pandas data frame.

        The data frame has the following columns:
            - example_column (int): example index
            - position (int): position within example (one-based)
            - group (str): group label, one of the following:
                - "TP": grammar position, important for model prediction
                - "FN": grammar position, not important for model prediction,
                - "FP": background position, important for model prediction,
                - "TN": background position, not important for model prediction
            - label (str): label of example, e.g., "cell type 1"
        """
        fi_matrix = results[0]
        y: List[str] = results[2]
        annotations: List[str] = results[3]

        example_column: List[int] = list()
        position_column: List[int] = list()
        group_column: List[str] = list()
        label_column: List[str] = list()

        for example_id, annotation in enumerate(annotations):
            example_column += [example_id] * len(annotation)
            position_column += list(range(1, len(annotation) + 1))
            group_column += [self.__get_agreement_group(
                char, fi_matrix[example_id, i, :])
                for i, char in enumerate(annotation)]
            label_column += [y[example_id]] * len(annotation)

        df = pd.DataFrame({"example": example_column,
                           "position": position_column,
                           "group": group_column,
                           "label": label_column})

        return df

    def _convert_to_unthresholded_data_frame(self, results) -> pd.DataFrame:
        """Takes gradient-based evaluator-specific results and turns them into
        a pandas data frame.

        The data frame has the following columns:
            - example_column (int): example index
            - position (int): position within example (one-based)
            - value (float): normalized feature importance
            - group (str): group label, one of the following:
                - "G": grammar position
                - "C": confounding position
                - "_": background position
            - label (str): label of example, e.g., "cell type 1"
        """
        fi_matrix = results[0]
        y: List[str] = results[2]
        annotations: List[str] = results[3]

        example_column: List[int] = list()
        position_column: List[int] = list()
        value_column: List[int] = list()
        group_column: List[str] = list()
        label_column: List[str] = list()

        for example_id, annotation in enumerate(annotations):
            example_column += [example_id] * len(annotation)
            position_column += list(range(1, len(annotation) + 1))
            fi_vector = GradientBasedEvaluator._prepare_normalized_fi_vector(
                fi_matrix[example_id, :, :])
            value_column += list(fi_vector)
            group_column += list(annotation)
            label_column += [y[example_id]] * len(annotation)

        df = pd.DataFrame({"example": example_column,
                           "position": position_column,
                           "value": value_column,
                           "group": group_column,
                           "label": label_column})

        return df


class GradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.GRADIENT, "Vanilla gradient saliency",
                         learner, output_dir)
        self.explainer = VanillaGradExplainer(learner.model)


class GradientxInputEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.GRADIENT_X_INPUT, "Gradient x input",
                         learner, output_dir)
        self.explainer = GradxInputExplainer(learner.model)


class SaliencyEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.SALIENCY, "Saliency", learner,
                         output_dir)
        self.explainer = SaliencyExplainer(learner.model)


class IntegratedGradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.INTEGRATED_GRADIENTS,
                         "Integrated Gradients", learner, output_dir)
        self.explainer = IntegrateGradExplainer(learner.model)


class NonlinearIntegratedGradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        # TODO NonlinearIntegratedGradExplainer
        # requires other data and how to handle reference (default is None)
        super().__init__(c.EvaluatorID.NONLINEAR_INTEGRATED_GRADIENTS,
                         "Nonlinear Integrated Gradients", learner,
                         output_dir)
        # self.explainer = NonlinearIntegrateGradExplainer(learner.model)


class GradCamGradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.GRAD_CAM, "Grad-CAM", learner,
                         output_dir)
        self.explainer = GradCAMExplainer(learner.model)

    def _explainer_transform(self, data, result):
        return torch.nn.functional.interpolate(result.view(1, 1, -1),
                                               size=data.shape[2],
                                               mode="linear").cpu().numpy()


class DeepLiftEvaluator(GradientBasedEvaluator):
    # TODO where to set reference?
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.DEEP_LIFT, "DeepLIFT", learner,
                         output_dir)
        self.explainer = DeepLIFTRescaleExplainer(learner.model, "shuffled")


class ExcitationBackpropEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.EXCITATION_BACKPROP,
                         "Excitation Backprop", learner, output_dir)
        self.explainer = ExcitationBackpropExplainer(learner.model)


class ContrastiveExcitationBackpropEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__(c.EvaluatorID.CONTRASTIVE_EXCITATION_BACKPROP,
                         "Contrastive Excitation Backprop", learner,
                         output_dir)
        self.explainer = ContrastiveExcitationBackpropExplainer(
            learner.model)
