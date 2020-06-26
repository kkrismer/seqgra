"""Gradient x Data Evaluator
"""
from typing import Optional

import seqgra.constants as c
from seqgra.learner import Learner
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator


class GradientxInputEvaluator(AbstractGradientEvaluator):
    """Gradient x Data evaluator for PyTorch models
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None) -> None:
        super().__init__(c.EvaluatorID.GRADIENT_X_INPUT, "Gradient x input",
                         learner, output_dir, importance_threshold)

    def explain(self, x, y):
        grad = self._backprop(x, y)
        return x.data * grad
