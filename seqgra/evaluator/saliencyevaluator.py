"""
MIT - CSAIL - Gifford Lab - seqgra

- class for all saliency evaluators
- models must be in pytorch

@author: Jennifer Hammelma
"""
import torch

from seqgra.learner import Learner
from seqgra.evaluator import Evaluator
from seqgra.evaluator.explainer.backprop import VanillaGradExplainer
from seqgra.evaluator.explainer.backprop import GradxInputExplainer
from seqgra.evaluator.explainer.backprop import SaliencyExplainer
from seqgra.evaluator.explainer.backprop import IntegrateGradExplainer
from seqgra.evaluator.explainer.backprop import NonlinearIntegrateGradExplainer
from seqgra.evaluator.explainer.deeplift import DeepLIFTRescaleExplainer
from seqgra.evaluator.explainer.gradcam import GradCAMExplainer
from seqgra.evaluator.explainer.ebp import ExcitationBackpropExplainer
from seqgra.evaluator.explainer.ebp import ContrastiveExcitationBackpropExplainer


class GradientBasedEvaluator(Evaluator):
    def __init__(self, id: str, learner: Learner, output_dir: str) -> None:
        super().__init__(id, learner, output_dir)
        self.explainer = None

    def evaluate_model(self, set_name: str = "test") -> None:
        '''
        TODO make calls to calculate saliency
        of form of SISEvaluator evaluate_model
        '''

    def calculate_saliency(self, data, label):
        result = self.explainer.explain(data, label)
        return self._explainer_transform(data, result)

    def _explainer_transform(self, data, result):
        return result.cuda().numpy()


class GradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("gradient", learner, output_dir)
        self.explainer = VanillaGradExplainer(learner.model)


class GradientxInputEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("gradientx-input", learner, output_dir)
        self.explainer = GradxInputExplainer(learner.model)


class SaliencyEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("saliency", learner, output_dir)
        self.explainer = SaliencyExplainer(learner.model)


class IntegratedGradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("integrated-gradient", learner, output_dir)
        self.explainer = IntegrateGradExplainer(learner.model)


class NonlinearIntegratedGradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        # TODO NonlinearIntegratedGradExplainer
        # requires other data and how to handle reference (default is None)
        super().__init__("nonlinear-integrated-gradient", learner,
                         output_dir)
        # self.explainer = NonlinearIntegrateGradExplainer(learner.model)


class GradCamGradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("grad-cam-gradient", learner, output_dir)
        self.explainer = GradCAMExplainer(learner.model)

    def _explainer_transform(self, data, result):
        return torch.nn.functional.interpolate(result.view(1, 1, -1),
                                               size=data.shape[2],
                                               mode="linear").cpu().numpy()


class DeepLiftEvaluator(GradientBasedEvaluator):
    # TODO where to set reference?
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("deep-lift", learner, output_dir)
        self.explainer = DeepLIFTRescaleExplainer(learner.model, "shuffled")


class ExcitationBackpropEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("excitation-backprop", learner, output_dir)
        self.explainer = ExcitationBackpropExplainer(learner.model)


class ContrastiveExcitationBackpropEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("contrastive-excitation-backprop", learner,
                         output_dir)
        self.explainer = ContrastiveExcitationBackpropExplainer(
            learner.model)
