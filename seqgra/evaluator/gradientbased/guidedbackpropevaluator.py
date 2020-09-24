"""Gradient Saliency Evaluator
"""
import types
from typing import Optional

import torch
from torch.autograd import Function

import seqgra.constants as c
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator
from seqgra.learner import Learner


class GuidedBackpropEvaluator(AbstractGradientEvaluator):
    """Guided backprop evaluator for PyTorch models
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 silent: bool = False) -> None:
        super().__init__(c.EvaluatorID.GUIDED_BACKPROP, "Guided Backprop",
                         learner, output_dir, importance_threshold,
                         silent=silent)
        self._override_backward()

    def explain(self, x, y):
        return self._backprop(x, y)

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                ctx.save_for_backward(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                output, = ctx.saved_tensors
                mask1 = (output > 0).float()
                mask2 = (grad_output.data > 0).float()
                grad_inp = mask1 * mask2 * grad_output.data
                grad_output.data.copy_(grad_inp)
                return grad_output

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.learner.model.apply(replace)
