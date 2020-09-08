"""Gradient Saliency Evaluator
"""
from typing import Optional

import seqgra.constants as c
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator
from seqgra.learner import Learner


class GradientEvaluator(AbstractGradientEvaluator):
    """Vanilla gradient saliency evaluator for PyTorch models
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 silent: bool = False) -> None:
        super().__init__(c.EvaluatorID.GRADIENT, "Vanilla gradient saliency",
                         learner, output_dir, importance_threshold,
                         silent=silent)

    def explain(self, x, y):
        return self._backprop(x, y)
