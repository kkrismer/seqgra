"""Smooth Grad Difference Evaluator
"""
from typing import Optional

import torch

import seqgra.constants as c
from seqgra.learner import Learner
from seqgra.evaluator.gradientbased import AbstractDifferenceGradientEvaluator


class SmoothGradEvaluator(AbstractDifferenceGradientEvaluator):
    """Smooth Grad difference gradient evaluator for PyTorch models

    modified from https://github.com/PAIR-code/saliency/blob/master/saliency/base.py#L80
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 stdev_spread: float = 0.15,
                 nsamples: int = 25, magnitude: bool = True,
                 silent: bool = False) -> None:
        super().__init__(c.EvaluatorID.SMOOTH_GRAD, "Smooth Grad", learner,
                         output_dir, importance_threshold, silent=silent)
        self.stdev_spread: float = stdev_spread
        self.nsamples: int = nsamples
        self.magnitude: bool = magnitude

    def explain(self, x, y1, y2):
        stdev = self.stdev_spread * (x.data.max() - x.data.min())

        total_gradients = 0
        origin_inp_data = x.data.clone()

        for i in range(self.nsamples):
            noise = torch.randn(x.size()).to(self.learner.device) * stdev
            x.data.copy_(noise + origin_inp_data)
            grad = self._backprop(x, y1, y2)

            if self.magnitude:
                total_gradients += grad ** 2
            else:
                total_gradients += grad

        return total_gradients / self.nsamples
