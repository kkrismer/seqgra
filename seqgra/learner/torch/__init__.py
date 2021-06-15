"""
seqgra learner hierarchy - PyTorch

Classes:
    - :class:`~seqgra.learner.torch.torchlearner.TorchDNAMultiClassClassificationLearner`: PyTorch implementation of DNA-based multi-class classification learner
    - :class:`~seqgra.learner.torch.torchlearner.TorchDNAMultiLabelClassificationLearner`: PyTorch implementation of DNA-based multi-label classification learner
    - :class:`~seqgra.learner.torch.torchlearner.TorchProteinMultiClassClassificationLearner`: PyTorch implementation of protein-based multi-class classification learner
    - :class:`~seqgra.learner.torch.torchlearner.TorchProteinMultiLabelClassificationLearner`: PyTorch implementation of protein-based multi-label classification learner
    - :class:`~seqgra.learner.torch.torchhelper.TorchHelper`: helper class for TensorFlow learners
"""
from seqgra.learner.torch.torchhelper import TorchHelper
from seqgra.learner.torch.torchlearner import TorchDNAMultiClassClassificationLearner
from seqgra.learner.torch.torchlearner import TorchDNAMultiLabelClassificationLearner
from seqgra.learner.torch.torchlearner import TorchProteinMultiClassClassificationLearner
from seqgra.learner.torch.torchlearner import TorchProteinMultiLabelClassificationLearner
