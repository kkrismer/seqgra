from typing import FrozenSet, List


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
    BAYES_OPTIMAL_CLASSIFIER: str = "BayesOptimalClassifier"
    ALL_LIBRARIES: FrozenSet[str] = frozenset([TENSORFLOW, TORCH,
                                               BAYES_OPTIMAL_CLASSIFIER])


class DataSet:
    TRAINING: str = "training"
    VALIDATION: str = "validation"
    TEST: str = "test"
    ALL_SETS: List[str] = list([TRAINING, VALIDATION, TEST])


class EvaluatorID:
    METRICS: str = "metrics"
    PREDICT: str = "predict"
    ROC: str = "roc"
    PR: str = "pr"
    SIS: str = "sis"
    GRADIENT: str = "gradient"
    GRADIENT_X_INPUT: str = "gradient-x-input"
    SALIENCY: str = "saliency"
    FEEDBACK: str = "feedback"
    GUIDED_BACKPROP: str = "guided-backprop"
    DECONV: str = "deconv"
    SMOOTH_GRAD: str = "smooth-grad"
    INTEGRATED_GRADIENTS: str = "integrated-gradients"
    NONLINEAR_INTEGRATED_GRADIENTS: str = "nonlinear-integrated-gradients"
    GRAD_CAM: str = "grad-cam"
    DEEP_LIFT: str = "deep-lift"
    EXCITATION_BACKPROP: str = "excitation-backprop"
    CONTRASTIVE_EXCITATION_BACKPROP: str = "contrastive-excitation-backprop"
    CONVENTIONAL_EVALUATORS: FrozenSet[str] = frozenset(
        [METRICS, PREDICT, ROC, PR])
    MODEL_AGNOSTIC_EVALUATORS: FrozenSet[str] = frozenset(
        [METRICS, PREDICT, ROC, PR, SIS])
    FEATURE_IMPORTANCE_EVALUATORS: FrozenSet[str] = frozenset(
        [SIS, GRADIENT, GRADIENT_X_INPUT, SALIENCY, FEEDBACK, GUIDED_BACKPROP,
         DECONV, SMOOTH_GRAD, INTEGRATED_GRADIENTS, 
         NONLINEAR_INTEGRATED_GRADIENTS, GRAD_CAM, DEEP_LIFT, 
         EXCITATION_BACKPROP, CONTRASTIVE_EXCITATION_BACKPROP])
    ALL_EVALUATOR_IDS: FrozenSet[str] = frozenset(
        [METRICS, PREDICT, ROC, PR, SIS, GRADIENT, GRADIENT_X_INPUT, SALIENCY,
         FEEDBACK, GUIDED_BACKPROP, DECONV, SMOOTH_GRAD, INTEGRATED_GRADIENTS,
         NONLINEAR_INTEGRATED_GRADIENTS, GRAD_CAM, DEEP_LIFT,
         EXCITATION_BACKPROP, CONTRASTIVE_EXCITATION_BACKPROP])

class ComparatorID:
    ROC: str = "roc"
    PR: str = "pr"
    ALL_COMPARATOR_IDS: FrozenSet[str] = frozenset([ROC, PR])