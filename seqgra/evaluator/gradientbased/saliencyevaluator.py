"""Gradient Saliency Evaluator
"""
from typing import Optional

import seqgra.constants as c
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator
from seqgra.learner import Learner


class SaliencyEvaluator(AbstractGradientEvaluator):
    """Gradient saliency evaluator for PyTorch models
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None) -> None:
        super().__init__(c.EvaluatorID.SALIENCY, "Saliency", learner,
                         output_dir, importance_threshold)

    def explain(self, x, y):
        grad = self._backprop(x, y)
        return grad.abs()
