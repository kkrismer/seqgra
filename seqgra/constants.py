from typing import FrozenSet


class PositionType:
    GRAMMAR: str = "G"
    BACKGROUND: str = "_"
    CONFOUNDER: str = "C"
    DNA_MASKED: str = "N"
    AA_MASKED: str = "X"


class TaskType:
    MULTI_CLASS_CLASSIFICATION: str = "multi-class classification"
    MULTI_LABEL_CLASSIFICATION: str = "multi-label classification"
    MULTIPLE_REGRESSION: str = "multiple regression"
    MULTIVARIATE_REGRESSION: str = "multivariate regression"
    ALL_TASKS: FrozenSet[str] = frozenset([MULTI_CLASS_CLASSIFICATION,
                                           MULTI_LABEL_CLASSIFICATION,
                                           MULTIPLE_REGRESSION,
                                           MULTIVARIATE_REGRESSION])


class SequenceSpaceType:
    DNA: str = "DNA"
    PROTEIN: str = "protein"
    ALL_SEQUENCE_SPACES: FrozenSet[str] = frozenset([DNA, PROTEIN])


class LibraryType:
    TENSORFLOW: str = "TensorFlow"
    TORCH: str = "PyTorch"
    ALL_LIBRARIES: FrozenSet[str] = frozenset([TENSORFLOW, TORCH])


class EvaluatorID:
    METRICS: str = "metrics"
    PREDICT: str = "predict"
    ROC: str = "roc"
    PR: str = "pr"
    SIS: str = "sis"
    GRADIENT: str = "gradient"
    GRADIENTX_INPUT: str = "gradientx-input"
    SALIENCY: str = "saliency"
    INTEGRATED_GRADIENT: str = "integrated-gradient"
    NONLINEAR_INTEGRATED_GRADIENT: str = "nonlinear-integrated-gradient"
    GRAD_CAM_GRADIENT: str = "grad-cam-gradient"
    DEEP_LIFT: str = "deep-lift"
    EXCITATION_BACKPROP: str = "excitation-backprop"
    CONTRASTIVE_EXCITATION_BACKPROP: str = "contrastive-excitation-backprop"
    CONVENTIONAL_EVALUATORS: FrozenSet[str] = frozenset(
        [METRICS, PREDICT, ROC, PR])
    MODEL_AGNOSTIC_EVALUATORS: FrozenSet[str] = frozenset(
        [METRICS, PREDICT, ROC, PR, SIS])
    FEATURE_IMPORTANCE_EVALUATORS: FrozenSet[str] = frozenset(
        [SIS, GRADIENT, GRADIENTX_INPUT, SALIENCY,
         INTEGRATED_GRADIENT, NONLINEAR_INTEGRATED_GRADIENT, GRAD_CAM_GRADIENT,
         DEEP_LIFT, EXCITATION_BACKPROP, CONTRASTIVE_EXCITATION_BACKPROP])
    ALL_EVALUATOR_IDS: FrozenSet[str] = frozenset(
        [METRICS, PREDICT, ROC, PR, SIS, GRADIENT, GRADIENTX_INPUT, SALIENCY,
         INTEGRATED_GRADIENT, NONLINEAR_INTEGRATED_GRADIENT, GRAD_CAM_GRADIENT,
         DEEP_LIFT, EXCITATION_BACKPROP, CONTRASTIVE_EXCITATION_BACKPROP])
