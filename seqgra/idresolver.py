"""
MIT - CSAIL - Gifford Lab - seqgra

Class with miscellaneous helper functions as static methods

@author: Konstantin Krismer
"""
import os
from typing import Optional

import seqgra.constants as c
from seqgra.evaluator import Evaluator
from seqgra.learner import Learner
from seqgra.learner.bayes import BayesOptimalDNAMultiClassClassificationLearner
from seqgra.learner.bayes import BayesOptimalDNAMultiLabelClassificationLearner
from seqgra.learner.bayes import BayesOptimalProteinMultiClassClassificationLearner
from seqgra.learner.bayes import BayesOptimalProteinMultiLabelClassificationLearner
from seqgra.model import DataDefinition
from seqgra.model import ModelDefinition

class IdResolver:
    @staticmethod
    def get_learner(model_definition: ModelDefinition,
                    data_definition: Optional[DataDefinition],
                    data_dir: str, output_dir: str,
                    validate_data: bool,
                    gpu_id: int,
                    silent: bool) -> Learner:
        if data_definition is not None:
            if model_definition.task != data_definition.task:
                raise Exception("model and grammar task incompatible (" +
                                "model task: " + model_definition.task +
                                ", grammar task: " + data_definition.task + ")")
            if model_definition.sequence_space != data_definition.sequence_space:
                raise Exception("model and grammar sequence space incompatible (" +
                                "model sequence space: " +
                                model_definition.sequence_space +
                                ", grammar sequence space: " +
                                data_definition.sequence_space + ")")

        if model_definition.library == c.LibraryType.TENSORFLOW:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

        # imports are inside if branches to only depend on TensorFlow and PyTorch
        # when required
        if model_definition.implementation is None:
            if model_definition.task == c.TaskType.MULTI_CLASS_CLASSIFICATION:
                if model_definition.sequence_space == c.SequenceSpaceType.DNA:
                    if model_definition.library == c.LibraryType.TENSORFLOW:
                        from seqgra.learner.tensorflow import KerasDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                        return KerasDNAMultiClassClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            gpu_id, silent)
                    elif model_definition.library == c.LibraryType.TORCH:
                        from seqgra.learner.torch import TorchDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                        return TorchDNAMultiClassClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            gpu_id, silent)
                    elif model_definition.library == c.LibraryType.BAYES_OPTIMAL_CLASSIFIER:
                        return BayesOptimalDNAMultiClassClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            silent)
                    else:
                        raise Exception("invalid library: " +
                                        model_definition.library)
                elif model_definition.sequence_space == c.SequenceSpaceType.PROTEIN:
                    if model_definition.library == c.LibraryType.TENSORFLOW:
                        from seqgra.learner.tensorflow import KerasProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                        return KerasProteinMultiClassClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            gpu_id, silent)
                    elif model_definition.library == c.LibraryType.TORCH:
                        from seqgra.learner.torch import TorchProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                        return TorchProteinMultiClassClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            gpu_id, silent)
                    elif model_definition.library == c.LibraryType.BAYES_OPTIMAL_CLASSIFIER:
                        return BayesOptimalProteinMultiClassClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            silent)
                    else:
                        raise Exception("invalid library: " +
                                        model_definition.library)
                else:
                    raise Exception("invalid model sequence space: " +
                                    model_definition.sequence_space)
            elif model_definition.task == c.TaskType.MULTI_LABEL_CLASSIFICATION:
                if model_definition.sequence_space == c.SequenceSpaceType.DNA:
                    if model_definition.library == c.LibraryType.TENSORFLOW:
                        from seqgra.learner.tensorflow import KerasDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                        return KerasDNAMultiLabelClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            gpu_id, silent)
                    elif model_definition.library == c.LibraryType.TORCH:
                        from seqgra.learner.torch import TorchDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                        return TorchDNAMultiLabelClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            gpu_id, silent)
                    elif model_definition.library == c.LibraryType.BAYES_OPTIMAL_CLASSIFIER:
                        return BayesOptimalDNAMultiLabelClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            silent)
                    else:
                        raise Exception("invalid library: " +
                                        model_definition.library)
                elif model_definition.sequence_space == c.SequenceSpaceType.PROTEIN:
                    if model_definition.library == c.LibraryType.TENSORFLOW:
                        from seqgra.learner.tensorflow import KerasProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                        return KerasProteinMultiLabelClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            gpu_id, silent)
                    elif model_definition.library == c.LibraryType.TORCH:
                        from seqgra.learner.torch import TorchProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                        return TorchProteinMultiLabelClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            gpu_id, silent)
                    elif model_definition.library == c.LibraryType.BAYES_OPTIMAL_CLASSIFIER:
                        return BayesOptimalProteinMultiLabelClassificationLearner(
                            model_definition, data_dir, output_dir, validate_data,
                            silent)
                    else:
                        raise Exception("invalid library: " +
                                        model_definition.library)
                else:
                    raise Exception("invalid model sequence space: " +
                                    model_definition.sequence_space)
            elif model_definition.task == c.TaskType.MULTIPLE_REGRESSION:
                raise NotImplementedError("implementation for multiple "
                                          "regression not available")
            elif model_definition.task == c.TaskType.MULTIVARIATE_REGRESSION:
                raise NotImplementedError("implementation for multivariate "
                                          "regression not available")
            else:
                raise Exception("invalid model task: " + model_definition.task)
        else:
            if model_definition.implementation == "KerasDNAMultiClassClassificationLearner":
                from seqgra.learner.tensorflow import KerasDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                return KerasDNAMultiClassClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, gpu_id,
                    silent)
            elif model_definition.implementation == "KerasDNAMultiLabelClassificationLearner":
                from seqgra.learner.tensorflow import KerasDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                return KerasDNAMultiLabelClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, gpu_id,
                    silent)
            elif model_definition.implementation == "TorchDNAMultiClassClassificationLearner":
                from seqgra.learner.torch import TorchDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                return TorchDNAMultiClassClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, gpu_id,
                    silent)
            elif model_definition.implementation == "TorchDNAMultiLabelClassificationLearner":
                from seqgra.learner.torch import TorchDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                return TorchDNAMultiLabelClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, gpu_id,
                    silent)
            elif model_definition.implementation == "BayesOptimalDNAMultiClassClassificationLearner":
                return BayesOptimalDNAMultiClassClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, silent)
            elif model_definition.implementation == "BayesOptimalDNAMultiLabelClassificationLearner":
                return BayesOptimalDNAMultiLabelClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, silent)
            elif model_definition.implementation == "KerasProteinMultiClassClassificationLearner":
                from seqgra.learner.tensorflow import KerasProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                return KerasProteinMultiClassClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, gpu_id,
                    silent)
            elif model_definition.implementation == "KerasProteinMultiLabelClassificationLearner":
                from seqgra.learner.tensorflow import KerasProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                return KerasProteinMultiLabelClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, gpu_id,
                    silent)
            elif model_definition.implementation == "TorchProteinMultiClassClassificationLearner":
                from seqgra.learner.torch import TorchProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                return TorchProteinMultiClassClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, gpu_id,
                    silent)
            elif model_definition.implementation == "TorchProteinMultiLabelClassificationLearner":
                from seqgra.learner.torch import TorchProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                return TorchProteinMultiLabelClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, gpu_id,
                    silent)
            elif model_definition.implementation == "BayesOptimalProteinMultiClassClassificationLearner":
                return BayesOptimalProteinMultiClassClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, silent)
            elif model_definition.implementation == "BayesOptimalProteinMultiLabelClassificationLearner":
                return BayesOptimalProteinMultiLabelClassificationLearner(
                    model_definition, data_dir, output_dir, validate_data, silent)
            else:
                raise Exception("invalid learner ID")

    @staticmethod
    def get_evaluator(evaluator_id: str, learner: Learner,
                      output_dir: str,
                      eval_sis_predict_threshold: Optional[float] = None,
                      eval_grad_importance_threshold: Optional[float] = None,
                      silent: bool = False) -> Evaluator:
        evaluator_id = evaluator_id.lower().strip()

        if learner is None:
            raise Exception("no learner specified")

        if evaluator_id == c.EvaluatorID.METRICS:
            from seqgra.evaluator import MetricsEvaluator  # pylint: disable=import-outside-toplevel
            return MetricsEvaluator(learner, output_dir, silent)
        elif evaluator_id == c.EvaluatorID.PREDICT:
            from seqgra.evaluator import PredictEvaluator  # pylint: disable=import-outside-toplevel
            return PredictEvaluator(learner, output_dir, silent)
        elif evaluator_id == c.EvaluatorID.ROC:
            from seqgra.evaluator import ROCEvaluator  # pylint: disable=import-outside-toplevel
            return ROCEvaluator(learner, output_dir, silent)
        elif evaluator_id == c.EvaluatorID.PR:
            from seqgra.evaluator import PREvaluator  # pylint: disable=import-outside-toplevel
            return PREvaluator(learner, output_dir, silent)
        elif evaluator_id == c.EvaluatorID.SIS:
            from seqgra.evaluator import SISEvaluator  # pylint: disable=import-outside-toplevel
            return SISEvaluator(learner, output_dir, eval_sis_predict_threshold,
                                silent=silent)
        elif evaluator_id == c.EvaluatorID.GRADIENT:
            from seqgra.evaluator.gradientbased import GradientEvaluator  # pylint: disable=import-outside-toplevel
            return GradientEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.GRADIENT_X_INPUT:
            from seqgra.evaluator.gradientbased import GradientxInputEvaluator  # pylint: disable=import-outside-toplevel
            return GradientxInputEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.SALIENCY:
            from seqgra.evaluator.gradientbased import SaliencyEvaluator  # pylint: disable=import-outside-toplevel
            return SaliencyEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.FEEDBACK:
            from seqgra.evaluator.gradientbased import FeedbackEvaluator  # pylint: disable=import-outside-toplevel
            return FeedbackEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.GUIDED_BACKPROP:
            from seqgra.evaluator.gradientbased import GuidedBackpropEvaluator  # pylint: disable=import-outside-toplevel
            return GuidedBackpropEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.DECONV:
            from seqgra.evaluator.gradientbased import DeconvEvaluator  # pylint: disable=import-outside-toplevel
            return DeconvEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.SMOOTH_GRAD:
            from seqgra.evaluator.gradientbased import SmoothGradEvaluator  # pylint: disable=import-outside-toplevel
            return SmoothGradEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.INTEGRATED_GRADIENTS:
            from seqgra.evaluator.gradientbased import IntegratedGradientEvaluator  # pylint: disable=import-outside-toplevel
            return IntegratedGradientEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.NONLINEAR_INTEGRATED_GRADIENTS:
            from seqgra.evaluator.gradientbased import NonlinearIntegratedGradientEvaluator  # pylint: disable=import-outside-toplevel
            return NonlinearIntegratedGradientEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.GRAD_CAM:
            from seqgra.evaluator.gradientbased import GradCamGradientEvaluator  # pylint: disable=import-outside-toplevel
            return GradCamGradientEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.DEEP_LIFT:
            from seqgra.evaluator.gradientbased import DeepLiftEvaluator  # pylint: disable=import-outside-toplevel
            return DeepLiftEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.EXCITATION_BACKPROP:
            from seqgra.evaluator.gradientbased import ExcitationBackpropEvaluator  # pylint: disable=import-outside-toplevel
            return ExcitationBackpropEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        elif evaluator_id == c.EvaluatorID.CONTRASTIVE_EXCITATION_BACKPROP:
            from seqgra.evaluator.gradientbased import ContrastiveExcitationBackpropEvaluator  # pylint: disable=import-outside-toplevel
            return ContrastiveExcitationBackpropEvaluator(
                learner, output_dir, eval_grad_importance_threshold, silent=silent)
        else:
            raise Exception("invalid evaluator ID")
