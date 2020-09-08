"""Gradient Difference Evaluator
"""
from typing import Optional

import seqgra.constants as c
from seqgra.learner import Learner
from seqgra.evaluator.gradientbased import AbstractDifferenceGradientEvaluator


class DifferenceGradientEvaluator(AbstractDifferenceGradientEvaluator):
    """Vanilla difference gradient evaluator for PyTorch models
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 silent: bool = False) -> None:
        super().__init__(c.EvaluatorID.GRADIENT,
                         "Vanilla difference gradient saliency", learner,
                         output_dir, importance_threshold, silent=silent)

    def explain(self, x, y1, y2):
        return self._backprop(x, y1, y2)
