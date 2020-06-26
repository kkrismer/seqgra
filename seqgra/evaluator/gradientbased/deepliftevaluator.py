"""DeepLIFT Evaluator
"""
import types
from typing import Optional

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import seqgra.constants as c
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator
from seqgra.learner import Learner


class DeepLiftEvaluator(AbstractGradientEvaluator):
    """DeepLIFT evaluator for PyTorch models
    """
    # TODO where to set reference?

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 baseline_type: str = "shuffled") -> None:
        super().__init__(c.EvaluatorID.DEEP_LIFT, "DeepLIFT", learner,
                         output_dir, importance_threshold)
        self._prepare_reference()
        assert(baseline_type in ["neutral", "zeros", "shuffled"])
        self.baseline_inp = None
        self.baseline_type = baseline_type
        self._override_backward()

    def explain(self, x, y):
        self._reset_preference()
        self._baseline_forward(x)

        grad = self._backprop(x, y)
        return x.data * grad

    def _prepare_reference(self):
        def init_refs(m):
            name = m.__class__.__name__
            if name.find("ReLU") != -1:
                m.ref_inp_list = []
                m.ref_out_list = []

        def ref_forward(self, x):
            self.ref_inp_list.append(x.data.clone())
            out = F.relu(x)
            self.ref_out_list.append(out.data.clone())
            return out

        def ref_replace(m):
            name = m.__class__.__name__
            if name.find("ReLU") != -1:
                m.forward = types.MethodType(ref_forward, m)

        self.learner.model.apply(init_refs)
        self.learner.model.apply(ref_replace)

    def _reset_preference(self):
        def reset_refs(m):
            name = m.__class__.__name__
            if name.find("ReLU") != -1:
                m.ref_inp_list = []
                m.ref_out_list = []

        self.learner.model.apply(reset_refs)

    def _baseline_forward(self, inp):
        if self.baseline_inp is None:
            self.baseline_inp = inp.data.clone()
            if self.baseline_type is "neutral":
                self.baseline_inp.fill_(0.25)
            elif self.baseline_type is "zeros":
                self.baseline_inp.fill_(0.0)
            elif self.baseline_type is "shuffled":
                self.baseline_inp = self.baseline_inp[:, :, torch.randperm(
                    self.baseline_inp.size()[2])]
            # TODO baseline_inp with shuffled k-mers??
            self.baseline_inp = Variable(self.baseline_inp)

        # get ref
        _ = self.learner.model(self.baseline_inp)

    def _override_backward(self):
        def new_backward(self, grad_out):
            ref_inp, inp = self.ref_inp_list
            ref_out, out = self.ref_out_list
            delta_out = out - ref_out
            delta_in = inp - ref_inp
            g1 = (delta_in.abs() > 1e-5).float() * grad_out * \
                delta_out / delta_in
            mask = ((ref_inp + inp) > 0).float()
            g2 = (delta_in.abs() <= 1e-5).float() * 0.5 * mask * grad_out

            return g1 + g2

        def backward_replace(m):
            name = m.__class__.__name__
            if name.find("ReLU") != -1:
                m.backward = types.MethodType(new_backward, m)

        self.learner.model.apply(backward_replace)
