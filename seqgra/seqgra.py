#!/usr/bin/env python

"""MIT - CSAIL - Gifford Lab - seqgra

seqgra complete pipeline:
1. generate data based on data definition (once), see run_simulator.py
2. train model on data (once), see run_learner.py
3. evaluate model performance with SIS, see run_sis.py

@author: Konstantin Krismer
"""
import argparse
import logging
import os
import shutil
from typing import List, Optional

import seqgra
import seqgra.constants as c
from seqgra import MiscHelper, IdResolver
from seqgra.evaluator import Evaluator
from seqgra.evaluator import FeatureImportanceEvaluator
from seqgra.learner import Learner
from seqgra.model import DataDefinition
from seqgra.model import ModelDefinition
from seqgra.parser import DataDefinitionParser
from seqgra.parser import XMLDataDefinitionParser
from seqgra.parser import ModelDefinitionParser
from seqgra.parser import XMLModelDefinitionParser
from seqgra.simulator import Simulator
from seqgra.simulator.heatmap import GrammarPositionHeatmap


def run_seqgra(data_config_file: Optional[str],
               data_folder: Optional[str],
               model_config_file: Optional[str],
               evaluator_ids: Optional[List[str]],
               output_dir: str,
               in_memory: bool,
               print_info: bool,
               silent: bool,
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
    if silent:
        logger.setLevel(os.environ.get("LOGLEVEL", "WARNING"))
    output_dir = MiscHelper.format_output_dir(output_dir.strip())
    new_data: bool = False
    new_model: bool = False

    if data_config_file is None:
        data_definition: Optional[DataDefinition] = None
        grammar_id = data_folder.strip()
        logger.info("loaded experimental data")
        GrammarPositionHeatmap.create(output_dir + "input/" + grammar_id,
                                      c.DataSet.TRAINING)
        GrammarPositionHeatmap.create(output_dir + "input/" + grammar_id,
                                      c.DataSet.VALIDATION)
        GrammarPositionHeatmap.create(output_dir + "input/" + grammar_id,
                                      c.DataSet.TEST)
    else:
        # generate synthetic data
        data_config = MiscHelper.read_config_file(data_config_file)
        data_def_parser: DataDefinitionParser = XMLDataDefinitionParser(
            data_config, silent)
        data_definition: DataDefinition = data_def_parser.get_data_definition()
        grammar_id: str = data_definition.grammar_id
        if print_info:
            print(data_definition)

        if remove_existing_data:
            simulator_output_dir: str = output_dir + "input/" + grammar_id
            if os.path.exists(simulator_output_dir):
                shutil.rmtree(simulator_output_dir, ignore_errors=True)
                logger.info("removed existing synthetic data")

        simulator = Simulator(data_definition, output_dir + "input", silent)
        synthetic_data_available: bool = \
            len(os.listdir(simulator.output_dir)) > 0
        if synthetic_data_available:
            logger.info("loaded previously generated synthetic data")
        else:
            logger.info("generating synthetic data")
            simulator.simulate_data()
            new_data = True

        simulator.create_grammar_heatmap(c.DataSet.TRAINING)
        simulator.create_grammar_heatmap(c.DataSet.VALIDATION)
        simulator.create_grammar_heatmap(c.DataSet.TEST)

        simulator.create_motif_info()
        simulator.create_motif_kl_divergence_matrix()
        simulator.create_empirical_similarity_score_matrix()

    # get learner
    if model_config_file is not None:
        model_config = MiscHelper.read_config_file(model_config_file)
        model_def_parser: ModelDefinitionParser = XMLModelDefinitionParser(
            model_config, silent)
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

        learner: Learner = IdResolver.get_learner(
            model_definition, data_definition,
            output_dir + "input/" + grammar_id,
            output_dir + "models/" + grammar_id,
            not no_checks, gpu_id, silent)

        # train model on data
        trained_model_available: bool = len(os.listdir(learner.output_dir)) > 0
        train_model: bool = not trained_model_available
        if trained_model_available:
            try:
                learner.load_model()
                logger.info("loaded previously trained model")
            except Exception as exception:
                logger.warning("unable to load previously trained model; "
                               "previously trained model will be deleted "
                               "and new model will be trained; "
                               "exception caught: %s", str(exception))

                # delete all files and folders in learner output directory
                for file_name in os.listdir(learner.output_dir):
                    file_path = os.path.join(learner.output_dir, file_name)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except OSError as e:
                        logger.warning(
                            "Failed to delete %s. Reason: %s", file_path, e)

                train_model = True

            if new_data and train_model:
                raise Exception("previously trained model used outdated "
                                "training data; delete '" +
                                learner.output_dir +
                                "' and run seqgra again to train new model "
                                "on current data")

        if train_model:
            logger.info("training model")

            learner.create_model()
            if print_info:
                learner.print_model_summary()

            if in_memory:
                training_set_file: str = learner.get_examples_file(
                    c.DataSet.TRAINING)
                validation_set_file: str = learner.get_examples_file(
                    c.DataSet.VALIDATION)
                x_train, y_train = learner.parse_examples_data(
                    training_set_file)
                x_val, y_val = learner.parse_examples_data(validation_set_file)

                learner.train_model(x_train=x_train, y_train=y_train,
                                    x_val=x_val, y_val=y_val)
            else:
                learner.train_model()
            learner.save_model()
            new_model = learner.definition.library != \
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
                    evaluator: Evaluator = IdResolver.get_evaluator(
                        evaluator_id, learner, evaluation_dir,
                        eval_sis_predict_threshold,
                        eval_grad_importance_threshold, silent)

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
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s " + seqgra.__version__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-d",
        "--data-config-file",
        type=str,
        help="path to the segra XML data configuration file. Use this option "
        "to generate synthetic data based on a seqgra grammar (specify "
        "either -d or -f, not both)"
    )
    group.add_argument(
        "-f",
        "--data-folder",
        type=str,
        help="experimental data folder name inside outputdir/input. Use this "
        "option to train the model on experimental or externally synthesized "
        "data (specify either -f or -d, not both)"
    )
    parser.add_argument(
        "-m",
        "--model-config-file",
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
        "--output-dir",
        type=str,
        required=True,
        help="output directory, subdirectories are created for generated "
        "data, trained model, and model evaluation"
    )
    parser.add_argument(
        "-i",
        "--in-memory",
        action="store_true",
        help="if this flag is set, training and validation data will be "
        "stored in-memory instead of loaded in chunks"
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="if this flag is set, data definition, model definition, and "
        "model summary are printed"
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="if this flag is set, only warnings and errors are printed"
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
        "--no-checks",
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

    if args.data_folder and args.model_config_file is None:
        parser.error("-f/--data-folder requires -m/--model-config-file.")

    if args.evaluators and args.model_config_file is None:
        parser.error("-e/--evaluators requires -m/--model-config-file.")

    if args.evaluators is not None:
        for evaluator in args.evaluators:
            if evaluator not in c.EvaluatorID.ALL_EVALUATOR_IDS:
                raise ValueError(
                    "invalid evaluator ID {s!r}".format(s=evaluator))

    run_seqgra(args.data_config_file,
               args.data_folder,
               args.model_config_file,
               args.evaluators,
               args.output_dir,
               args.in_memory,
               args.print,
               args.silent,
               args.remove,
               args.gpu,
               args.no_checks,
               args.eval_sets,
               args.eval_n,
               args.eval_n_per_label,
               args.eval_suppress_plots,
               args.eval_fi_predict_threshold,
               args.eval_sis_predict_threshold,
               args.eval_grad_importance_threshold)


if __name__ == "__main__":
    main()
