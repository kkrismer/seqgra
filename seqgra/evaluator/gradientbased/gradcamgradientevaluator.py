"""GradCAM Evaluator
"""
from typing import Optional

import torch

import seqgra.constants as c
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator
from seqgra.learner import Learner


class GradCamGradientEvaluator(AbstractGradientEvaluator):
    """GradCAM Evaluator
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 target_layer_name_keys=None, use_input: bool = False) -> None:
        super().__init__(c.EvaluatorID.GRAD_CAM, "Grad-CAM", learner,
                         output_dir, importance_threshold)
        self.target_layer = self.get_layer(target_layer_name_keys)
        self.use_input = use_input
        self.intermediate_act = []
        self.intermediate_grad = []
        self._register_forward_backward_hook()

    def _explainer_transform(self, data, result):
        return torch.nn.functional.interpolate(result.view(1, 1, -1),
                                               size=data.shape[2],
                                               mode="linear").cpu().numpy()

    def _register_forward_backward_hook(self):
        def forward_hook_input(m, i, o):
            self.intermediate_act.append(i[0].data.clone())

        def forward_hook_output(m, i, o):
            self.intermediate_act.append(o.data.clone())

        def backward_hook(m, grad_i, grad_o):
            self.intermediate_grad.append(grad_o[0].data.clone())

        if self.use_input:
            self.target_layer.register_forward_hook(forward_hook_input)
        else:
            self.target_layer.register_forward_hook(forward_hook_output)

        self.target_layer.register_backward_hook(backward_hook)

    def _reset_intermediate_lists(self):
        self.intermediate_act = []
        self.intermediate_grad = []

    def explain(self, x, y):
        self._reset_intermediate_lists()

        _ = self._backprop(x, y)

        grad = self.intermediate_grad[0]
        act = self.intermediate_act[0]

        weights = grad.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
        cam = weights * act
        cam = cam.sum(1).unsqueeze(1)

        cam = torch.clamp(cam, min=0)

        return cam
