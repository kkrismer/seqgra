#!/usr/bin/env python

"""MIT - CSAIL - Gifford Lab - seqgra

seqgra attribution

@author: Konstantin Krismer
"""

import argparse
import logging
import os
import shutil
from typing import List

import seqgra
from seqgra import AnnotatedExampleSet
from seqgra import MiscHelper
import seqgra.constants as c
from seqgra.evaluator import Evaluator
from seqgra.idresolver import IdResolver
from seqgra.learner import Learner
from seqgra.model import ModelDefinition
from seqgra.parser import ModelDefinitionParser
from seqgra.parser import XMLModelDefinitionParser


def obtain_feature_attribution(attr_dir: str, grammar_id: str,
                               learner: Learner, evaluator_id: str,
                               examples_file: str, annotations_file: str,
                               target: str,
                               eval_sis_predict_threshold: float,
                               eval_grad_importance_threshold: float,
                               eval_suppress_plots: bool) -> None:
    logger = logging.getLogger(__name__)

    evaluation_dir: str = MiscHelper.prepare_path(
        attr_dir + "/" + grammar_id + "/" +
        learner.definition.model_id,
        allow_exists=True, allow_non_empty=True)

    if len(os.listdir(evaluation_dir)) > 0:
        logger.info("skip evaluator %s: results already saved "
                    "to disk", evaluator_id)
    else:
        logger.info("running feature importance "
                    "evaluator %s on selected examples", evaluator_id)

        evaluator: Evaluator = IdResolver.get_evaluator(
            evaluator_id, learner, evaluation_dir,
            eval_sis_predict_threshold,
            eval_grad_importance_threshold)

        learner.set_seed()

        x, y, annotations = load_data(learner, examples_file, annotations_file)

        if x:
            try:
                # TODO target
                results = evaluator._evaluate_model(x, y, annotations)
                evaluator._save_results(results, "custom", eval_suppress_plots)
                if not eval_suppress_plots and results is not None:
                    evaluator._visualize_grammar_agreement(results, "custom")
            except RuntimeError as err:
                with open(evaluation_dir + "custom-incompatible-model.txt",
                          "w") as incompatible_model_file:
                    incompatible_model_file.write(
                        "evaluator skipped: model incompatible\n")
                    incompatible_model_file.write("error thrown:\n")
                    incompatible_model_file.write(str(err))

                logger.warning("evaluator skipped: incompatible with "
                               "model (error thrown: %s)", err)

        else:
            with open(evaluation_dir + "custom-no-valid-examples.txt",
                      "w") as no_examples_file:
                no_examples_file.write("no valid examples\n")

            logger.warning("evaluator skipped: no valid examples")


def load_data(learner: Learner, examples_file: str,
              annotations_file: str) -> AnnotatedExampleSet:
    x, y = learner.parse_examples_data(examples_file)
    annotations, _ = learner.parse_annotations_data(annotations_file)

    return AnnotatedExampleSet(x, y, annotations)


def prepare_model(grammar_id: str, model_def_file: str,
                  output_dir: str, gpu_id: int) -> Learner:
    logger = logging.getLogger(__name__)

    model_config = MiscHelper.read_config_file(model_def_file)
    model_def_parser: ModelDefinitionParser = XMLModelDefinitionParser(
        model_config, True)
    model_definition: ModelDefinition = model_def_parser.get_model_definition()

    learner: Learner = IdResolver.get_learner(
        model_definition, None,
        output_dir + "input/" + grammar_id,
        output_dir + "models/" + grammar_id,
        False, gpu_id, True)

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

    if train_model:
        logger.info("training model")

        learner.create_model()
        learner.train_model()
        learner.save_model()

    return learner


def run_seqgra_attribution(analysis_id: str,
                           grammar_ids: List[str],
                           model_def_files: List[str],
                           output_dir: str,
                           examples_file: str,
                           annotations_file: str,
                           evaluator_ids: List[str],
                           target: str,
                           eval_sis_predict_threshold: float,
                           eval_grad_importance_threshold: float,
                           eval_suppress_plots: bool,
                           gpu_id: int) -> None:
    logger = logging.getLogger(__name__)

    analysis_id = MiscHelper.sanitize_id(analysis_id)
    output_dir = MiscHelper.format_output_dir(output_dir.strip())
    attr_dir: str = MiscHelper.prepare_path(
        output_dir + "attribution/" + analysis_id,
        allow_exists=True, allow_non_empty=True)

    # get learner
    for grammar_id in grammar_ids:
        for model_def_file in model_def_files:
            learner: Learner = prepare_model(grammar_id, model_def_file,
                                             output_dir, gpu_id)

            for evaluator_id in evaluator_ids:
                logger.info(
                    "evaluate model %s trained on %s with evaluator %s",
                    learner.definition.model_id, grammar_id, evaluator_id)
                obtain_feature_attribution(attr_dir, grammar_id, learner,
                                           evaluator_id,
                                           examples_file, annotations_file,
                                           target,
                                           eval_sis_predict_threshold,
                                           eval_grad_importance_threshold,
                                           eval_suppress_plots)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="seqgraa",
        description="seqgra attribution: Obtain feature attribution/evidence "
        "for selected examples across multiple grammars, models, evaluators")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s " + seqgra.__version__)
    parser.add_argument(
        "-a",
        "--analysis-id",
        type=str,
        required=True,
        help="analysis ID (used for script file name and comparator folders)"
    )
    parser.add_argument(
        "-d",
        "--grammar-ids",
        type=str,
        required=True,
        nargs="+",
        help="list of grammar IDs or data folders (inside output-dir/input)"
    )
    parser.add_argument(
        "-m",
        "--model-def-files",
        type=str,
        required=True,
        nargs="+",
        help="list of paths to the seqgra XML model definition files"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="output directory of previous seqgra calls with subdirectories "
        "input (simulated and experimental data) and models (trained models); "
        "subdirectories are created for attribution analysis"
    )
    parser.add_argument(
        "-i",
        "--examples-file",
        type=str,
        required=True,
        help="path to file containing examples for feature importance "
        "evaluators"
    )
    parser.add_argument(
        "-j",
        "--annotations-file",
        type=str,
        required=True,
        help="path to file containing annotations of examples for feature "
        "importance evaluators"
    )
    parser.add_argument(
        "-e",
        "--evaluators",
        type=str,
        default=c.EvaluatorID.CORE_FEATURE_IMPORTANCE_EVALUATORS,
        nargs="+",
        help="feature importance evaluator ID or IDs: " +
        ", ".join(sorted(c.EvaluatorID.FEATURE_IMPORTANCE_EVALUATORS))
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="y",
        help="target label for which evidence/attribution should be "
        "generated, either 'y' for ground truth label or 'y-hat' for "
        "predicted label, defaults to 'y'"
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
    parser.add_argument(
        "--eval-suppress-plots",
        action="store_true",
        help="if this flag is set, plots are suppressed globally; "
        "this evaluator argument will be passed to all evaluators"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
        help="ID of GPU used by TensorFlow and PyTorch (defaults to GPU "
        "ID 0); CPU is used if no GPU is available or GPU ID is set to -1"
    )

    return parser

def main():
    logging.basicConfig(level=logging.INFO)

    parser = create_parser()
    args = parser.parse_args()

    if args.target != "y" and args.target != "y-hat":
        raise ValueError("invalid value for -t/--target (only 'y' and "
                         "'y-hat' allowed)")

    if args.evaluators is not None:
        for evaluator in args.evaluators:
            if evaluator not in c.EvaluatorID.FEATURE_IMPORTANCE_EVALUATORS:
                raise ValueError(
                    "invalid FI evaluator ID {s!r}".format(s=evaluator))

    run_seqgra_attribution(args.analysis_id,
                           args.grammar_ids,
                           args.model_def_files,
                           args.output_dir,
                           args.examples_file,
                           args.annotations_file,
                           args.evaluators,
                           args.target,
                           args.eval_sis_predict_threshold,
                           args.eval_grad_importance_threshold,
                           args.eval_suppress_plots,
                           args.gpu)


if __name__ == "__main__":
    main()
