"""Integrated Gradient Evaluator
"""
from typing import Optional

import numpy as np
from torch.autograd import Variable

import seqgra.constants as c
from seqgra.learner import Learner
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator


class IntegratedGradientEvaluator(AbstractGradientEvaluator):
    """Integrated gradient evaluator for PyTorch models
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 steps: int = 100,
                 silent: bool = False) -> None:
        super().__init__(c.EvaluatorID.INTEGRATED_GRADIENTS,
                         "Integrated Gradients", learner, output_dir,
                         importance_threshold, silent=silent)
        self.steps = steps

    def explain(self, x, y):
        grad = 0
        x_data = x.data.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_x = Variable(x_data * alpha, requires_grad=True)
            g = self._backprop(new_x, y)
            grad += g

        return grad * x_data / self.steps
