"""Abstract Gradient Evaluator
"""
from abc import abstractmethod
from typing import Optional

from seqgra.evaluator.gradientbased import GradientBasedEvaluator
from seqgra.learner import Learner


class AbstractGradientEvaluator(GradientBasedEvaluator):
    """Abstract base class for gradient evaluators

    Only PyTorch models supported.
    """

    @abstractmethod
    def __init__(self, evaluator_id: str, evaluator_name: str,
                 learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None) -> None:
        super().__init__(evaluator_id, evaluator_name, learner, output_dir,
                         importance_threshold)

    @abstractmethod
    def explain(self, x, y):
        pass
    
    def _backprop(self, inp, ind):
        output = self.learner.model(inp)
        if ind is None:
            ind = output.data.max(1)[1].unsqueeze(0)
        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind, 1.0)
        output.backward(grad_out)
        return inp.grad.data
