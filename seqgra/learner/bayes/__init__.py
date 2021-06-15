"""
seqgra learner hierarchy - Bayes Optimal Classifier (BOC)

Classes:
    - :class:`~seqgra.learner.bayes.bayeslearner.BayesOptimalDNAMultiClassClassificationLearner`: BOC implementation of DNA-based multi-class classification learner
    - :class:`~seqgra.learner.bayes.bayeslearner.BayesOptimalDNAMultiLabelClassificationLearner`: BOC implementation of DNA-based multi-label classification learner
    - :class:`~seqgra.learner.bayes.bayeslearner.BayesOptimalProteinMultiClassClassificationLearner`: BOC implementation of protein-based multi-class classification learner
    - :class:`~seqgra.learner.bayes.bayeslearner.BayesOptimalProteinMultiLabelClassificationLearner`: BOC implementation of protein-based multi-label classification learner
    - :class:`~seqgra.learner.bayes.bayeshelper.BayesOptimalHelper`: helper class for Bayes Optimal Classifier learners
"""
from seqgra.learner.bayes.bayeshelper import BayesOptimalHelper
from seqgra.learner.bayes.bayeslearner import BayesOptimalDNAMultiClassClassificationLearner
from seqgra.learner.bayes.bayeslearner import BayesOptimalDNAMultiLabelClassificationLearner
from seqgra.learner.bayes.bayeslearner import BayesOptimalProteinMultiClassClassificationLearner
from seqgra.learner.bayes.bayeslearner import BayesOptimalProteinMultiLabelClassificationLearner
