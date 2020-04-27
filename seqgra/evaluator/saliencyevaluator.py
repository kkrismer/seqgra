"""
MIT - CSAIL - Gifford Lab - seqgra

- class for all saliency evaluators
- models must be in pytorch

@author: Jennifer Hammelman
"""
from typing import List, Optional, Any

import numpy as np
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
    def __init__(self, evaluator_id: str, learner: Learner, output_dir: str) -> None:
        super().__init__(evaluator_id, learner, output_dir)
        self.explainer = None

    def _evaluate_model(self, x: List[str], y: List[str]) -> Any:
        '''
        TODO make calls to calculate saliency
        of form of SISEvaluator evaluate_model
        '''
        use_cuda = torch.cuda.is_available()

        # encode
        encoded_x = self.learner.encode_x(x)
        encoded_y = self.learner.encode_y(y)

        # convert bool to float32 and long, as expected by explainers
        encoded_x = encoded_x.astype(np.float32)
        encoded_y = encoded_y.astype(np.int64)

        # add H dimension?
        # e.g., for 100 nt window, 1 example, TensorFlow format (N, H, W, C):
        # (1, 100, 4) -> (1, 1, 100, 4)
        encoded_x = np.expand_dims(encoded_x, axis=0)

        # TensorFlow format to Torch format (N, C, H, W):
        # e.g., for 100 nt window, 1 example, added height dimension:
        # (1, 1, 100, 4) -> (1, 4, 1, 100)
        encoded_x = np.transpose(encoded_x, (1, 3, 0, 2))

        # convert np array to torch tensor
        encoded_x = torch.from_numpy(encoded_x)
        encoded_y = torch.from_numpy(encoded_y)

        if use_cuda:
            # store input tensor, label tensor and model on GPU
            encoded_x = encoded_x.cuda()
            encoded_y = encoded_y.cuda()
            self.explainer.model.cuda()

        encoded_x = torch.autograd.Variable(encoded_x, requires_grad=True)

        # enable inference mode
        self.explainer.model.eval()

        print("final sizes:")
        print("x: " + str(encoded_x.size()))
        print("y: " + str(encoded_y.size()))
        result = self.calculate_saliency(encoded_x, encoded_y)

        print(type(result))
        print(result.shape)
        return result
        
    def _save_results(self, results, set_name: str = "test") -> None:
        pass

    def calculate_saliency(self, data, label):
        result = self.explainer.explain(data, label)
        return self._explainer_transform(data, result)

    def _explainer_transform(self, data, result):
        return result.cpu().numpy()


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
