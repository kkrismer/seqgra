"""
seqgra evaluator hierarchy

Classes:
    - :class:`~seqgra.evaluator.evaluator.Evaluator`: abstract base class for all evaluators
    - :class:`~seqgra.evaluator.evaluator.FeatureImportanceEvaluator`: abstract class for feature importance evaluators
    - :class:`~seqgra.evaluator.metricsevaluator.MetricsEvaluator`: metrics evaluator
    - :class:`~seqgra.evaluator.predictevaluator.PredictEvaluator`: predict evaluator
    - :class:`~seqgra.evaluator.prevaluator.PREvaluator`: PR curve evaluator
    - :class:`~seqgra.evaluator.rocevaluator.ROCEvaluator`: ROC curve evaluator
    - :class:`~seqgra.evaluator.sisevaluator.SISEvaluator`: SIS feature importance evaluator
"""
from seqgra.evaluator.evaluator import Evaluator
from seqgra.evaluator.evaluator import FeatureImportanceEvaluator
from seqgra.evaluator.metricsevaluator import MetricsEvaluator
from seqgra.evaluator.predictevaluator import PredictEvaluator
from seqgra.evaluator.prevaluator import PREvaluator
from seqgra.evaluator.rocevaluator import ROCEvaluator
from seqgra.evaluator.sisevaluator import SISEvaluator
