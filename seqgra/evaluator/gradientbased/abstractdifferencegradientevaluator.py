"""Abstract Difference Gradient Evaluator
"""
from abc import abstractmethod
from typing import Optional

import torch

from seqgra.evaluator.gradientbased import GradientBasedEvaluator
from seqgra.learner import Learner


class AbstractDifferenceGradientEvaluator(GradientBasedEvaluator):
    """Abstract base class for difference gradient evaluators

    Only PyTorch models supported.
    """

    @abstractmethod
    def __init__(self, evaluator_id: str, evaluator_name: str,
                 learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 silent: bool = False) -> None:
        super().__init__(evaluator_id, evaluator_name, learner, output_dir,
                         importance_threshold, silent=silent)

    @abstractmethod
    def explain(self, x, y1, y2):
        pass

    def _backprop(self, x, y1, y2=None):
        output = self.learner.model(x)
        if y1 is None:
            y1 = output.data.max(1)[1]
        if y2 is None:
            index0 = torch.LongTensor([0]).to(self.learner.device)
            y2 = output.data.topk(k=2, sorted=True)[1][0][0].unsqueeze(0)

        grad_out1 = output.data.clone()
        grad_out1.fill_(0.0)

        grad_out2 = output.data.clone()
        grad_out2.fill_(0.0)

        grad_out1.scatter_(1, y1.unsqueeze(0).t(), 1.0)
        grad_out2.scatter_(1, y2.unsqueeze(0).t(), 1.0)
        output.backward(grad_out1 - grad_out2)
        return x.grad.data
