"""Deconvolution Evaluator
"""
import types
from typing import Optional

import torch
from torch.autograd import Function

import seqgra.constants as c
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator
from seqgra.learner import Learner


class DeconvEvaluator(AbstractGradientEvaluator):
    """Deconvolution evaluator for PyTorch models
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None) -> None:
        super().__init__(c.EvaluatorID.DECONV, "Deconvolution",
                         learner, output_dir, importance_threshold)
        self._override_backward()

    def explain(self, x, y):
        return self._backprop(x, y)

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                grad_inp = torch.clamp(grad_output, min=0)
                return grad_inp

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.learner.model.apply(replace)