from typing import Set


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
    ALL_TASKS: Set[str] = set([MULTI_CLASS_CLASSIFICATION,
                               MULTI_LABEL_CLASSIFICATION,
                               MULTIPLE_REGRESSION,
                               MULTIVARIATE_REGRESSION])


class SequenceSpaceType:
    DNA: str = "DNA"
    PROTEIN: str = "protein"
    ALL_SEQUENCE_SPACES: Set[str] = set([DNA, PROTEIN])


class LibraryType:
    TENSORFLOW: str = "TensorFlow"
    TORCH: str = "PyTorch"
    ALL_LIBRARIES: Set[str] = set([TENSORFLOW, TORCH])
