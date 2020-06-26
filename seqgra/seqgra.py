#!/usr/bin/env python

'''
MIT - CSAIL - Gifford Lab - seqgra

seqgra complete pipeline:
1. generate data based on data definition (once), see run_simulator.py
2. train model on data (once), see run_learner.py
3. evaluate model performance with SIS, see run_sis.py

@author: Konstantin Krismer
'''

import argparse
import logging
import os
import shutil
from typing import List, Optional

import seqgra.constants as c
from seqgra.evaluator import Evaluator
from seqgra.evaluator import FeatureImportanceEvaluator
from seqgra.learner import Learner
from seqgra.learner.bayes import BayesOptimalDNAMultiClassClassificationLearner
from seqgra.learner.bayes import BayesOptimalDNAMultiLabelClassificationLearner
from seqgra.learner.bayes import BayesOptimalProteinMultiClassClassificationLearner
from seqgra.learner.bayes import BayesOptimalProteinMultiLabelClassificationLearner
from seqgra.model import DataDefinition
from seqgra.model import ModelDefinition
from seqgra.parser import DataDefinitionParser
from seqgra.parser import XMLDataDefinitionParser
from seqgra.parser import ModelDefinitionParser
from seqgra.parser import XMLModelDefinitionParser
from seqgra.simulator import Simulator


def read_config_file(file_name: str) -> str:
    with open(file_name.strip()) as f:
        config: str = f.read()
    return config


def get_learner(model_definition: ModelDefinition,
                data_definition: Optional[DataDefinition],
                data_dir: str, output_dir: str,
                validate_data: bool,
                gpu_id: int) -> Learner:
    if data_definition is not None:
        if model_definition.task != data_definition.task:
            raise Exception("model and grammar task incompatible (" +
                            "model task: " + model_definition.task +
                            ", grammar task: " + data_definition.task + ")")
        if model_definition.sequence_space != data_definition.sequence_space:
            raise Exception("model and grammar sequence space incompatible (" +
                            "model sequence space: " +
                            model_definition.sequence_space +
                            ", grammar sequence space: " +
                            data_definition.sequence_space + ")")

    if model_definition.library == c.LibraryType.TENSORFLOW:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    # imports are inside if branches to only depend on TensorFlow and PyTorch
    # when required
    if model_definition.implementation is None:
        if model_definition.task == c.TaskType.MULTI_CLASS_CLASSIFICATION:
            if model_definition.sequence_space == c.SequenceSpaceType.DNA:
                if model_definition.library == c.LibraryType.TENSORFLOW:
                    from seqgra.learner.tensorflow import KerasDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                    return KerasDNAMultiClassClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data,
                        gpu_id)
                elif model_definition.library == c.LibraryType.TORCH:
                    from seqgra.learner.torch import TorchDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                    return TorchDNAMultiClassClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data,
                        gpu_id)
                elif model_definition.library == c.LibraryType.BAYES_OPTIMAL_CLASSIFIER:
                    return BayesOptimalDNAMultiClassClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data)
                else:
                    raise Exception("invalid library: " +
                                    model_definition.library)
            elif model_definition.sequence_space == c.SequenceSpaceType.PROTEIN:
                if model_definition.library == c.LibraryType.TENSORFLOW:
                    from seqgra.learner.tensorflow import KerasProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                    return KerasProteinMultiClassClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data,
                        gpu_id)
                elif model_definition.library == c.LibraryType.TORCH:
                    from seqgra.learner.torch import TorchProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
                    return TorchProteinMultiClassClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data,
                        gpu_id)
                elif model_definition.library == c.LibraryType.BAYES_OPTIMAL_CLASSIFIER:
                    return BayesOptimalProteinMultiClassClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data)
                else:
                    raise Exception("invalid library: " +
                                    model_definition.library)
            else:
                raise Exception("invalid model sequence space: " +
                                model_definition.sequence_space)
        elif model_definition.task == c.TaskType.MULTI_LABEL_CLASSIFICATION:
            if model_definition.sequence_space == c.SequenceSpaceType.DNA:
                if model_definition.library == c.LibraryType.TENSORFLOW:
                    from seqgra.learner.tensorflow import KerasDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                    return KerasDNAMultiLabelClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data,
                        gpu_id)
                elif model_definition.library == c.LibraryType.TORCH:
                    from seqgra.learner.torch import TorchDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                    return TorchDNAMultiLabelClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data,
                        gpu_id)
                elif model_definition.library == c.LibraryType.BAYES_OPTIMAL_CLASSIFIER:
                    return BayesOptimalDNAMultiLabelClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data)
                else:
                    raise Exception("invalid library: " +
                                    model_definition.library)
            elif model_definition.sequence_space == c.SequenceSpaceType.PROTEIN:
                if model_definition.library == c.LibraryType.TENSORFLOW:
                    from seqgra.learner.tensorflow import KerasProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                    return KerasProteinMultiLabelClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data,
                        gpu_id)
                elif model_definition.library == c.LibraryType.TORCH:
                    from seqgra.learner.torch import TorchProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
                    return TorchProteinMultiLabelClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data,
                        gpu_id)
                elif model_definition.library == c.LibraryType.BAYES_OPTIMAL_CLASSIFIER:
                    return BayesOptimalProteinMultiLabelClassificationLearner(
                        model_definition, data_dir, output_dir, validate_data)
                else:
                    raise Exception("invalid library: " +
                                    model_definition.library)
            else:
                raise Exception("invalid model sequence space: " +
                                model_definition.sequence_space)
        elif model_definition.task == c.TaskType.MULTIPLE_REGRESSION:
            raise NotImplementedError("implementation for multiple "
                                      "regression not available")
        elif model_definition.task == c.TaskType.MULTIVARIATE_REGRESSION:
            raise NotImplementedError("implementation for multivariate "
                                      "regression not available")
        else:
            raise Exception("invalid model task: " + model_definition.task)
    else:
        if model_definition.implementation == "KerasDNAMultiClassClassificationLearner":
            from seqgra.learner.tensorflow import KerasDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
            return KerasDNAMultiClassClassificationLearner(
                model_definition, data_dir, output_dir, validate_data, gpu_id)
        elif model_definition.implementation == "KerasDNAMultiLabelClassificationLearner":
            from seqgra.learner.tensorflow import KerasDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
            return KerasDNAMultiLabelClassificationLearner(
                model_definition, data_dir, output_dir, validate_data, gpu_id)
        elif model_definition.implementation == "TorchDNAMultiClassClassificationLearner":
            from seqgra.learner.torch import TorchDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
            return TorchDNAMultiClassClassificationLearner(
                model_definition, data_dir, output_dir, validate_data, gpu_id)
        elif model_definition.implementation == "TorchDNAMultiLabelClassificationLearner":
            from seqgra.learner.torch import TorchDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
            return TorchDNAMultiLabelClassificationLearner(
                model_definition, data_dir, output_dir, validate_data, gpu_id)
        elif model_definition.implementation == "BayesOptimalDNAMultiClassClassificationLearner":
            return BayesOptimalDNAMultiClassClassificationLearner(
                model_definition, data_dir, output_dir, validate_data)
        elif model_definition.implementation == "BayesOptimalDNAMultiLabelClassificationLearner":
            return BayesOptimalDNAMultiLabelClassificationLearner(
                model_definition, data_dir, output_dir, validate_data)
        elif model_definition.implementation == "KerasProteinMultiClassClassificationLearner":
            from seqgra.learner.tensorflow import KerasProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
            return KerasProteinMultiClassClassificationLearner(
                model_definition, data_dir, output_dir, validate_data, gpu_id)
        elif model_definition.implementation == "KerasProteinMultiLabelClassificationLearner":
            from seqgra.learner.tensorflow import KerasProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
            return KerasProteinMultiLabelClassificationLearner(
                model_definition, data_dir, output_dir, validate_data, gpu_id)
        elif model_definition.implementation == "TorchProteinMultiClassClassificationLearner":
            from seqgra.learner.torch import TorchProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
            return TorchProteinMultiClassClassificationLearner(
                model_definition, data_dir, output_dir, validate_data, gpu_id)
        elif model_definition.implementation == "TorchProteinMultiLabelClassificationLearner":
            from seqgra.learner.torch import TorchProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
            return TorchProteinMultiLabelClassificationLearner(
                model_definition, data_dir, output_dir, validate_data, gpu_id)
        elif model_definition.implementation == "BayesOptimalProteinMultiClassClassificationLearner":
            return BayesOptimalProteinMultiClassClassificationLearner(
                model_definition, data_dir, output_dir, validate_data)
        elif model_definition.implementation == "BayesOptimalProteinMultiLabelClassificationLearner":
            return BayesOptimalProteinMultiLabelClassificationLearner(
                model_definition, data_dir, output_dir, validate_data)
        else:
            raise Exception("invalid learner ID")


def get_evaluator(evaluator_id: str, learner: Learner,
                  output_dir: str,
                  eval_sis_predict_threshold: Optional[float] = None,
                  eval_grad_importance_threshold: Optional[float] = None) -> Evaluator:
    evaluator_id = evaluator_id.lower().strip()

    if learner is None:
        raise Exception("no learner specified")

    if evaluator_id == c.EvaluatorID.METRICS:
        from seqgra.evaluator import MetricsEvaluator  # pylint: disable=import-outside-toplevel
        return MetricsEvaluator(learner, output_dir)
    elif evaluator_id == c.EvaluatorID.PREDICT:
        from seqgra.evaluator import PredictEvaluator  # pylint: disable=import-outside-toplevel
        return PredictEvaluator(learner, output_dir)
    elif evaluator_id == c.EvaluatorID.ROC:
        from seqgra.evaluator import ROCEvaluator  # pylint: disable=import-outside-toplevel
        return ROCEvaluator(learner, output_dir)
    elif evaluator_id == c.EvaluatorID.PR:
        from seqgra.evaluator import PREvaluator  # pylint: disable=import-outside-toplevel
        return PREvaluator(learner, output_dir)
    elif evaluator_id == c.EvaluatorID.SIS:
        from seqgra.evaluator import SISEvaluator  # pylint: disable=import-outside-toplevel
        return SISEvaluator(learner, output_dir, eval_sis_predict_threshold)
    elif evaluator_id == c.EvaluatorID.GRADIENT:
        from seqgra.evaluator.gradientbased import GradientEvaluator  # pylint: disable=import-outside-toplevel
        return GradientEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.GRADIENT_X_INPUT:
        from seqgra.evaluator.gradientbased import GradientxInputEvaluator  # pylint: disable=import-outside-toplevel
        return GradientxInputEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.SALIENCY:
        from seqgra.evaluator.gradientbased import SaliencyEvaluator  # pylint: disable=import-outside-toplevel
        return SaliencyEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.FEEDBACK:
        from seqgra.evaluator.gradientbased import FeedbackEvaluator  # pylint: disable=import-outside-toplevel
        return FeedbackEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.GUIDED_BACKPROP:
        from seqgra.evaluator.gradientbased import GuidedBackpropEvaluator  # pylint: disable=import-outside-toplevel
        return GuidedBackpropEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.DECONV:
        from seqgra.evaluator.gradientbased import DeconvEvaluator  # pylint: disable=import-outside-toplevel
        return DeconvEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.SMOOTH_GRAD:
        from seqgra.evaluator.gradientbased import SmoothGradEvaluator  # pylint: disable=import-outside-toplevel
        return SmoothGradEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.INTEGRATED_GRADIENTS:
        from seqgra.evaluator.gradientbased import IntegratedGradientEvaluator  # pylint: disable=import-outside-toplevel
        return IntegratedGradientEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.NONLINEAR_INTEGRATED_GRADIENTS:
        from seqgra.evaluator.gradientbased import NonlinearIntegratedGradientEvaluator  # pylint: disable=import-outside-toplevel
        return NonlinearIntegratedGradientEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.GRAD_CAM:
        from seqgra.evaluator.gradientbased import GradCamGradientEvaluator  # pylint: disable=import-outside-toplevel
        return GradCamGradientEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.DEEP_LIFT:
        from seqgra.evaluator.gradientbased import DeepLiftEvaluator  # pylint: disable=import-outside-toplevel
        return DeepLiftEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.EXCITATION_BACKPROP:
        from seqgra.evaluator.gradientbased import ExcitationBackpropEvaluator  # pylint: disable=import-outside-toplevel
        return ExcitationBackpropEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    elif evaluator_id == c.EvaluatorID.CONTRASTIVE_EXCITATION_BACKPROP:
        from seqgra.evaluator.gradientbased import ContrastiveExcitationBackpropEvaluator  # pylint: disable=import-outside-toplevel
        return ContrastiveExcitationBackpropEvaluator(
            learner, output_dir, eval_grad_importance_threshold)
    else:
        raise Exception("invalid evaluator ID")


def format_output_dir(output_dir: str) -> str:
    output_dir = output_dir.strip().replace("\\", "/")
    if not output_dir.endswith("/"):
        output_dir += "/"
    return output_dir


def get_valid_file(data_file: str) -> str:
    data_file = data_file.replace("\\", "/").replace("//", "/").strip()
    if os.path.isfile(data_file):
        return data_file
    else:
        raise Exception("file does not exist: " + data_file)


def run_seqgra(data_config_file: Optional[str],
               data_folder: Optional[str],
               model_config_file: Optional[str],
               evaluator_ids: Optional[List[str]],
               output_dir: str,
               print_info: bool,
               remove_existing_data: bool,
               gpu_id: int,
               no_checks: bool,
               eval_sets: Optional[List[str]],
               eval_n: Optional[int],
               eval_n_per_label: Optional[int],
               eval_suppress_plots: Optional[bool],
               eval_fi_predict_threshold: Optional[float],
               eval_sis_predict_threshold: Optional[float],
               eval_grad_importance_threshold: Optional[float]) -> None:
    logger = logging.getLogger(__name__)
    output_dir = format_output_dir(output_dir.strip())
    new_data: bool = False
    new_model: bool = False

    if data_config_file is None:
        data_definition: Optional[DataDefinition] = None
        grammar_id = data_folder.strip()
        logger.info("loaded experimental data")
    else:
        # generate synthetic data
        data_config = read_config_file(data_config_file)
        data_def_parser: DataDefinitionParser = XMLDataDefinitionParser(
            data_config)
        data_definition: DataDefinition = data_def_parser.get_data_definition()
        grammar_id: str = data_definition.grammar_id
        if print_info:
            print(data_definition)

        if remove_existing_data:
            simulator_output_dir: str = output_dir + "input/" + grammar_id
            if os.path.exists(simulator_output_dir):
                shutil.rmtree(simulator_output_dir, ignore_errors=True)
                logger.info("removed existing synthetic data")

        simulator = Simulator(data_definition, output_dir + "input")
        synthetic_data_available: bool = \
            len(os.listdir(simulator.output_dir)) > 0
        if synthetic_data_available:
            logger.info("loaded previously generated synthetic data")
        else:
            logger.info("generating synthetic data")
            simulator.simulate_data()
            new_data = True

    # get learner
    if model_config_file is not None:
        model_config = read_config_file(model_config_file)
        model_def_parser: ModelDefinitionParser = XMLModelDefinitionParser(
            model_config)
        model_definition: ModelDefinition = \
            model_def_parser.get_model_definition()
        if print_info:
            print(model_definition)

        if remove_existing_data:
            learner_output_dir: str = output_dir + "models/" + grammar_id + \
                "/" + model_definition.model_id
            if os.path.exists(learner_output_dir):
                shutil.rmtree(learner_output_dir, ignore_errors=True)
                logger.info("removed pretrained model")

        learner: Learner = get_learner(model_definition, data_definition,
                                       output_dir + "input/" + grammar_id,
                                       output_dir + "models/" + grammar_id,
                                       not no_checks, gpu_id)

        # train model on data
        trained_model_available: bool = len(os.listdir(learner.output_dir)) > 0
        if trained_model_available:
            if new_data:
                raise Exception("previously trained model used outdated "
                                "training data; delete '" +
                                learner.output_dir +
                                "' and run seqgra again to train new model "
                                "on current data")
            learner.load_model()
            logger.info("loaded previously trained model")
        else:
            logger.info("training model")

            # load data
            training_set_file: str = learner.get_examples_file(
                c.DataSet.TRAINING)
            validation_set_file: str = learner.get_examples_file(
                c.DataSet.VALIDATION)
            x_train, y_train = learner.parse_examples_data(training_set_file)
            x_val, y_val = learner.parse_examples_data(validation_set_file)

            learner.create_model()
            if print_info:
                learner.print_model_summary()
            learner.train_model(x_train=x_train, y_train=y_train,
                                x_val=x_val, y_val=y_val)
            learner.save_model()
            new_model = True and learner.definition.library != \
                c.LibraryType.BAYES_OPTIMAL_CLASSIFIER

        if remove_existing_data:
            evaluator_output_dir: str = output_dir + "evaluation/" + \
                grammar_id + "/" + model_definition.model_id
            if os.path.exists(evaluator_output_dir):
                shutil.rmtree(evaluator_output_dir, ignore_errors=True)
                logger.info("removed evaluator results")

        if evaluator_ids:
            logger.info("evaluating model using interpretability methods")

            if eval_sets:
                for eval_set in eval_sets:
                    if not eval_set in c.DataSet.ALL_SETS:
                        raise Exception(
                            "invalid set selected for evaluation: " +
                            eval_set + "; only the following sets are "
                            "allowed: " +
                            ", ".join(c.DataSet.ALL_SETS))
            else:
                eval_sets: List[str] = c.DataSet.ALL_SETS

            evaluation_dir: str = output_dir + "evaluation/" + \
                grammar_id + "/" + learner.definition.model_id

            for evaluator_id in evaluator_ids:
                results_dir: str = evaluation_dir + "/" + evaluator_id
                results_exist: bool = os.path.exists(results_dir) and \
                    len(os.listdir(results_dir)) > 0
                if results_exist:
                    logger.info("skip evaluator %s: results already saved "
                                "to disk", evaluator_id)
                    if new_model:
                        logger.warning("results from evaluator %s are based "
                                       "on an outdated model; delete "
                                       "'%s' and run seqgra again to get "
                                       "results from %s on current model",
                                       evaluator_id, results_dir,
                                       evaluator_id)
                else:
                    evaluator: Evaluator = get_evaluator(
                        evaluator_id, learner, evaluation_dir,
                        eval_sis_predict_threshold,
                        eval_grad_importance_threshold)

                    if eval_n_per_label:
                        eval_n = eval_n_per_label

                    for eval_set in eval_sets:
                        learner.set_seed()

                        is_fi_evaluator: bool = isinstance(
                            evaluator, FeatureImportanceEvaluator)
                        if is_fi_evaluator:
                            logger.info("running feature importance "
                                        "evaluator %s on %s set",
                                        evaluator_id, eval_set)
                        else:
                            eval_fi_predict_threshold = None
                            logger.info("running evaluator %s on %s set",
                                        evaluator_id, eval_set)

                        evaluator.evaluate_model(
                            eval_set,
                            subset_n=eval_n,
                            subset_n_per_label=eval_n_per_label is not None,
                            subset_threshold=eval_fi_predict_threshold,
                            suppress_plots=eval_suppress_plots)
        else:
            logger.info("skipping evaluation step: no evaluator specified")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="seqgra",
        description="Generate synthetic data based on grammar, train model on "
        "synthetic data, evaluate model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-d",
        "--dataconfigfile",
        type=str,
        help="path to the segra XML data configuration file. Use this option "
        "to generate synthetic data based on a seqgra grammar (specify "
        "either -d or -f, not both)"
    )
    group.add_argument(
        "-f",
        "--datafolder",
        type=str,
        help="experimental data folder name inside outputdir/input. Use this "
        "option to train the model on experimental or externally synthesized "
        "data (specify either -f or -d, not both)"
    )
    parser.add_argument(
        "-m",
        "--modelconfigfile",
        type=str,
        help="path to the seqgra XML model configuration file"
    )
    parser.add_argument(
        "-e",
        "--evaluators",
        type=str,
        default=None,
        nargs="+",
        help="evaluator ID or IDs: IDs of "
        "conventional evaluators include " +
        ", ".join(sorted(c.EvaluatorID.CONVENTIONAL_EVALUATORS)) +
        "; IDs of feature importance evaluators include " +
        ", ".join(sorted(c.EvaluatorID.FEATURE_IMPORTANCE_EVALUATORS))
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        type=str,
        required=True,
        help="output directory, subdirectories are created for generated "
        "data, trained model, and model evaluation"
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="if this flag is set, data definition, model definition, and "
        "model summary are printed"
    )
    parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
        help="if this flag is set, previously stored data for this grammar - "
        "model combination will be removed prior to the analysis run. This "
        "includes the folders input/[grammar ID], "
        "models/[grammar ID]/[model ID], and "
        "evaluation/[grammar ID]/[model ID]."
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
        help="ID of GPU used by TensorFlow and PyTorch (defaults to GPU "
        "ID 0); CPU is used if no GPU is available or GPU ID is set to -1"
    )
    parser.add_argument(
        "--nochecks",
        action="store_true",
        help="if this flag is set, examples and example annotations will not "
        "be validated before training, e.g., that DNA sequences only contain "
        "A, C, G, T, N"
    )
    parser.add_argument(
        "--eval-sets",
        type=str,
        default=[c.DataSet.TEST],
        nargs="+",
        help="either one or more of the following: training, validation, "
        "test; selects data set for evaluation; this evaluator argument "
        "will be passed to all evaluators"
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        help="maximum number of examples to be evaluated per set (defaults "
        "to the total number of examples); this evaluator argument "
        "will be passed to all evaluators"
    )
    parser.add_argument(
        "--eval-n-per-label",
        type=int,
        help="maximum number of examples to be evaluated for each label and "
        "set (defaults to the total number of examples unless eval-n is set, "
        "overrules eval-n); "
        "this evaluator argument will be passed to all evaluators"
    )
    parser.add_argument(
        "--eval-suppress-plots",
        action="store_true",
        help="if this flag is set, plots are suppressed globally; "
        "this evaluator argument will be passed to all evaluators"
    )
    parser.add_argument(
        "--eval-fi-predict-threshold",
        type=float,
        default=0.5,
        help="prediction threshold used to select examples for evaluation, "
        "only examples with predict(x) > threshold will be passed on to "
        "evaluators (defaults to 0.5); "
        "this evaluator argument will be passed to feature importance "
        "evaluators only"
    )
    parser.add_argument(
        "--eval-sis-predict-threshold",
        type=float,
        default=0.5,
        help="prediction threshold for Sufficient Input Subsets; "
        "this evaluator argument is only visible to the SIS evaluator"
    )
    parser.add_argument(
        "--eval-grad-importance-threshold",
        type=float,
        default=0.01,
        help="feature importance threshold for gradient-based feature "
        "importance evaluators; this parameter only affects thresholded "
        "grammar agreement plots, not the feature importance measures "
        "themselves; this evaluator argument is only visible to "
        "gradient-based feature importance evaluators (defaults to 0.01)"
    )
    args = parser.parse_args()

    if args.datafolder and args.modelconfigfile is None:
        parser.error("-f/--datafolder requires -m/--modelconfigfile.")

    if args.evaluators and args.modelconfigfile is None:
        parser.error("-e/--evaluators requires -m/--modelconfigfile.")

    if args.evaluators is not None:
        for evaluator in args.evaluators:
            if evaluator not in c.EvaluatorID.ALL_EVALUATOR_IDS:
                raise ValueError(
                    "invalid evaluator ID {s!r}".format(s=evaluator))

    run_seqgra(args.dataconfigfile,
               args.datafolder,
               args.modelconfigfile,
               args.evaluators,
               args.outputdir,
               args.print,
               args.remove,
               args.gpu,
               args.nochecks,
               args.eval_sets,
               args.eval_n,
               args.eval_n_per_label,
               args.eval_suppress_plots,
               args.eval_fi_predict_threshold,
               args.eval_sis_predict_threshold,
               args.eval_grad_importance_threshold)


if __name__ == "__main__":
    main()
