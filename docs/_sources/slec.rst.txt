Simulators, Learners, Evaluators, Comparators
=============================================

====================  ==============================================  ==================================  ===========  =================  =============
ID                    Class Name                                      Base Class                          Type         Part of command    Description
====================  ==============================================  ==================================  ===========  =================  =============
*none*                Simulator                                       *none*                              Simulator    ``seqgra``         generates synthetic data
*none*                Learner                                         *none*                              Learner      ``seqgra``         trains models on synthetic or experimental data
*none*                MultiClassClassificationLearner                 Learner                             Learner      ``seqgra``         learner for multi-class classification tasks
*none*                MultiLabelClassificationLearner                 Learner                             Learner      ``seqgra``         learner for multi-label classification tasks
*none*                DNAMultiClassClassificationLearner              MultiClassClassificationLearner     Learner      ``seqgra``         learner for DNA sequence multi-class classification tasks
*none*                DNAMultiLabelClassificationLearner              MultiLabelClassificationLearner     Learner      ``seqgra``         learner for DNA sequence multi-label classification tasks
*none*                KerasDNAMultiClassClassificationLearner         DNAMultiClassClassificationLearner  Learner      ``seqgra``         TensorFlow Keras learner for DNA sequence multi-class classification tasks 
*none*                TorchDNAMultiClassClassificationLearner         DNAMultiClassClassificationLearner  Learner      ``seqgra``         PyTorch learner for DNA sequence multi-class classification tasks
*none*                BayesOptimalDNAMultiClassClassificationLearner  DNAMultiClassClassificationLearner  Learner      ``seqgra``         BOC learner for DNA sequence multi-class classification tasks
*none*                KerasDNAMultiLabelClassificationLearner         DNAMultiLabelClassificationLearner  Learner      ``seqgra``         TensorFlow Keras learner for DNA sequence multi-label classification tasks
*none*                TorchDNAMultiLabelClassificationLearner         DNAMultiLabelClassificationLearner  Learner      ``seqgra``         PyTorch learner for DNA sequence multi-label classification tasks
*none*                BayesOptimalDNAMultiLabelClassificationLearner  DNAMultiLabelClassificationLearner  Learner      ``seqgra``         BOC learner for DNA sequence multi-label classification tasks
*none*                Evaluator                                       *none*                              Evaluator    ``seqgra``         evaluates trained models
*none*                FeatureImportanceEvaluator                      Evaluator                           Evaluator    ``seqgra``         evaluates trained models based on input feature importance/attribution/contribution (FI)
*none*                GradientBasedEvaluator                          FeatureImportanceEvaluator          Evaluator    ``seqgra``         evaluates trained models based on input FI, requires access to gradient
roc                   ROCEvaluator                                    Evaluator                           Evaluator    ``seqgra``         create ROC curves for trained models
pr                    PREvaluator                                     Evaluator                           Evaluator    ``seqgra``         create precision-recall curves for trained models
metrics               MetricsEvaluator                                Evaluator                           Evaluator    ``seqgra``         saves accuracy and loss of best model to file
prediction            PredictEvaluator                                Evaluator                           Evaluator    ``seqgra``         saves all example predictions to file
sis                   SISEvaluator                                    FeatureImportanceEvaluator          Evaluator    ``seqgra``         Sufficient Input Subsets (SIS) FI evaluator
gradient              GradientEvaluator                               GradientBasedEvaluator              Evaluator    ``seqgra``         raw gradient FI evaluator
saliency              SaliencyEvaluator                               GradientBasedEvaluator              Evaluator    ``seqgra``         absolute gradient FI evaluator
gradient-x-input      GradientxInputEvaluator                         GradientBasedEvaluator              Evaluator    ``seqgra``         gradient times input FI evaluator
integrated-gradients  IntegratedGradientEvaluator                     GradientBasedEvaluator              Evaluator    ``seqgra``         Integrated Gradients FI evaluator
guided-backprop       GuidedBackpropEvaluator                         GradientBasedEvaluator              Evaluator    ``seqgra``         Guided Backpropagation FI evaluator
deconv                DeconvEvaluator                                 GradientBasedEvaluator              Evaluator    ``seqgra``         Deconvolution FI evaluator
*none*                Comparator                                      *none*                              Comparator   ``seqgras``        compares properties and performance metrics across data sets, models
table                 TableComparator                                 Comparator                          Comparator   ``seqgras``        compiles table of properties from simulation processes, training processes, and evaluators
curve-table           CurveTableComparator                            Comparator                          Comparator   ``seqgras``        compiles table of ROC and PR curves
fi-eval-table         FIETableComparator                              Comparator                          Comparator   ``seqgras``        compiles table of metrics from FI evaluators
roc                   ROCComparator                                   Comparator                          Comparator   ``seqgras``        creates plot of ROC curves from multiple models
pr                    PRComparator                                    Comparator                          Comparator   ``seqgras``        creates plot of precision-recall curves from multiple models
====================  ==============================================  ==================================  ===========  =================  =============
