"""
seqgra schema and miscellaneous helper functions

Classes:
    - :class:`~seqgra.schema.AnnotatedExample`: (x, y, a) tuple - sequence window, label, annotation (data class)
    - :class:`~seqgra.schema.AnnotatedExampleSet`: (x, y, a) tuple - sequence windows, labels, annotations (named tuple)
    - :class:`~seqgra.schema.AnnotationSet`: (y, a) tuple - labels, annotations (named tuple)
    - :class:`~seqgra.schema.Example`: (x, y) tuple - sequence window, label (data class)
    - :class:`~seqgra.schema.ExampleSet`: (x, y) tuple - sequence windows, labels (named tuple)
    - :class:`~seqgra.schema.Metrics`: (loss, accuracy) tuple (named tuple)
    - :class:`~seqgra.schema.ModelSize`: (n1, n2) tuple - number of trainable parameters, number of non-trainable parameters (named tuple)
    - :class:`~seqgra.schema.ProbabilisticToken`: (t, p) tuple - token, probability (named tuple)
    - :class:`~seqgra.schema.MiscHelper`: miscellaneous helper functions
"""
from seqgra.schema import AnnotatedExample
from seqgra.schema import AnnotatedExampleSet
from seqgra.schema import AnnotationSet
from seqgra.schema import Example
from seqgra.schema import ExampleSet
from seqgra.schema import Metrics
from seqgra.schema import ModelSize
from seqgra.schema import ProbabilisticToken
from seqgra.misc import MiscHelper
__version__ = "0.0.3"
