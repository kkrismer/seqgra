"""Contrastive Excitation Backpropagation Evaluator
"""
import types
from typing import Optional

import torch

import seqgra.constants as c
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator
from seqgra.evaluator.gradientbased.ebphelper import EBConv2d, EBLinear, EBAvgPool2d
from seqgra.learner import Learner

class ContrastiveExcitationBackpropEvaluator(AbstractGradientEvaluator):
    """Contrastive excitation backpropagation evaluator for PyTorch models
    """
    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 intermediate_layer_keys=None,
                 output_layer_keys=None, final_linear_keys=None) -> None:
        super().__init__(c.EvaluatorID.CONTRASTIVE_EXCITATION_BACKPROP,
                         "Contrastive Excitation Backprop", learner,
                         output_dir, importance_threshold)
        self.intermediate_layer = self.get_layer(intermediate_layer_keys)
        self.output_layer = self.get_layer(output_layer_keys)
        self.final_linear = self.get_layer(final_linear_keys)
        self._override_backward()
        self._register_hooks()
        self.intermediate_vars = []

    def _override_backward(self):
        def new_linear(self, x):
            return EBLinear.apply(x, self.weight, self.bias)

        def new_conv2d(self, x):
            return EBConv2d.apply(x, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)

        def new_avgpool2d(self, x):
            return EBAvgPool2d.apply(x, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, 
                                     self.count_include_pad)

        def replace(m):
            name = m.__class__.__name__
            if name == "Linear":
                m.forward = types.MethodType(new_linear, m)
            elif name == "Conv2d":
                m.forward = types.MethodType(new_conv2d, m)
            elif name == "AvgPool2d":
                m.forward = types.MethodType(new_avgpool2d, m)

        self.learner.model.apply(replace)

    def _register_hooks(self):
        self.intermediate_vars = []

        def forward_hook(m, i, o):
            self.intermediate_vars.append(o)

        self.intermediate_layer.register_forward_hook(forward_hook)
        self.output_layer.register_forward_hook(forward_hook)

    def explain(self, x, y):
        self.intermediate_vars = []

        output = self.learner.model(x)
        output_var, intermediate_var = self.intermediate_vars

        if y is None:
            y = output.data.max(1)[1]
        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, y.unsqueeze(0).t(), 1.0)

        self.final_linear.weight.data *= -1.0
        neg_map_var = torch.autograd.grad(
            output, intermediate_var, grad_out, retain_graph=True)
        neg_map = neg_map_var[0].data.clone()

        self.final_linear.weight.data *= -1.0
        pos_map_var = torch.autograd.grad(
            output, intermediate_var, grad_out, retain_graph=True)
        pos_map = pos_map_var[0].data.clone()

        diff = pos_map - neg_map
        attmap_var = torch.autograd.grad(
            intermediate_var, output_var, diff, retain_graph=True)

        attmap = attmap_var[0].data.clone()
        attmap = torch.clamp(attmap.sum(1).unsqueeze(1), min=0.0)

        return attmap
