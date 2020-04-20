#!/usr/bin/env python

'''
MIT - CSAIL - Gifford Lab - seqgra

seqgra complete pipeline:
1. generate data based on data definition (once), see run_simulator.py
2. train model on data (once), see run_learner.py
3. evaluate model performance with SIS, see run_sis.py

@author: Konstantin Krismer
'''

import os
import argparse
import logging
from typing import List

from seqgra.parser.dataparser import DataParser
from seqgra.parser.xmldataparser import XMLDataParser
from seqgra.parser.modelparser import ModelParser
from seqgra.parser.xmlmodelparser import XMLModelParser
from seqgra.simulator.simulator import Simulator
from seqgra.learner.learner import Learner
from seqgra.evaluator.evaluator import Evaluator


def parse_config_file(file_name: str) -> str:
    with open(file_name.strip()) as f:
        config: str = f.read()
    return config


def get_learner(model_parser: ModelParser, data_parser_type: str,
                data_dir: str, output_dir: str) -> Learner:
    if data_parser_type is not None and \
       model_parser.get_learner_type() != data_parser_type:
        raise Exception("learner and data type incompatible (" +
                        "learner type: " + model_parser.get_learner_type() +
                        ", data type: " + data_parser_type + ")")

    # imports are inside if branches to only depend on TensorFlow and PyTorch
    # when required
    if model_parser.get_learner_implementation() == "KerasDNAMultiClassClassificationLearner":
        from seqgra.learner.keraslearner import KerasDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
        return KerasDNAMultiClassClassificationLearner(model_parser, data_dir, output_dir)
    elif model_parser.get_learner_implementation() == "KerasDNAMultiLabelClassificationLearner":
        from seqgra.learner.keraslearner import KerasDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
        return KerasDNAMultiLabelClassificationLearner(model_parser, data_dir, output_dir)
    elif model_parser.get_learner_implementation() == "TorchDNAMultiClassClassificationLearner":
        from seqgra.learner.torchlearner import TorchDNAMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
        return TorchDNAMultiClassClassificationLearner(model_parser, data_dir, output_dir)
    elif model_parser.get_learner_implementation() == "TorchDNAMultiLabelClassificationLearner":
        from seqgra.learner.torchlearner import TorchDNAMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
        return TorchDNAMultiLabelClassificationLearner(model_parser, data_dir, output_dir)
    elif model_parser.get_learner_implementation() == "KerasProteinMultiClassClassificationLearner":
        from seqgra.learner.keraslearner import KerasProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
        return KerasProteinMultiClassClassificationLearner(model_parser, data_dir, output_dir)
    elif model_parser.get_learner_implementation() == "KerasProteinMultiLabelClassificationLearner":
        from seqgra.learner.keraslearner import KerasProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
        return KerasProteinMultiLabelClassificationLearner(model_parser, data_dir, output_dir)
    elif model_parser.get_learner_implementation() == "TorchProteinMultiClassClassificationLearner":
        from seqgra.learner.torchlearner import TorchProteinMultiClassClassificationLearner  # pylint: disable=import-outside-toplevel
        return TorchProteinMultiClassClassificationLearner(model_parser, data_dir, output_dir)
    elif model_parser.get_learner_implementation() == "TorchProteinMultiLabelClassificationLearner":
        from seqgra.learner.torchlearner import TorchProteinMultiLabelClassificationLearner  # pylint: disable=import-outside-toplevel
        return TorchProteinMultiLabelClassificationLearner(model_parser, data_dir, output_dir)
    else:
        raise Exception("invalid learner ID")


def get_evaluator(evaluator_id: str, learner: Learner,
                  output_dir: str) -> Evaluator:
    evaluator_id = evaluator_id.lower().strip()

    if learner is None:
        raise Exception("no learner specified")

    if evaluator_id == "metrics":
        from seqgra.evaluator.metricsevaluator import MetricsEvaluator  # pylint: disable=import-outside-toplevel
        return MetricsEvaluator(learner, output_dir)
    elif evaluator_id == "predict":
        from seqgra.evaluator.predictevaluator import PredictEvaluator  # pylint: disable=import-outside-toplevel
        return PredictEvaluator(learner, output_dir)
    elif evaluator_id == "roc":
        from seqgra.evaluator.rocevaluator import ROCEvaluator  # pylint: disable=import-outside-toplevel
        return ROCEvaluator(learner, output_dir)
    elif evaluator_id == "pr":
        from seqgra.evaluator.prevaluator import PREvaluator  # pylint: disable=import-outside-toplevel
        return PREvaluator(learner, output_dir)
    elif evaluator_id == "sis":
        from seqgra.evaluator.sisevaluator import SISEvaluator  # pylint: disable=import-outside-toplevel
        return SISEvaluator(learner, output_dir)
    elif evaluator_id == "gradient":
        from seqgra.evaluator.saliencyevaluator import GradientEvaluator  # pylint: disable=import-outside-toplevel
        return GradientEvaluator(learner, output_dir)
    elif evaluator_id == "gradientx-input":
        from seqgra.evaluator.saliencyevaluator import GradientxInputEvaluator  # pylint: disable=import-outside-toplevel
        return GradientxInputEvaluator(learner, output_dir)
    elif evaluator_id == "saliency":
        from seqgra.evaluator.saliencyevaluator import SaliencyEvaluator  # pylint: disable=import-outside-toplevel
        return SaliencyEvaluator(learner, output_dir)
    elif evaluator_id == "integrated-gradient":
        from seqgra.evaluator.saliencyevaluator import IntegratedGradientEvaluator  # pylint: disable=import-outside-toplevel
        return IntegratedGradientEvaluator(learner, output_dir)
    elif evaluator_id == "nonlinear-integrated-gradient":
        from seqgra.evaluator.saliencyevaluator import NonlinearIntegratedGradientEvaluator  # pylint: disable=import-outside-toplevel
        return NonlinearIntegratedGradientEvaluator(learner, output_dir)
    elif evaluator_id == "grad-cam-gradient":
        from seqgra.evaluator.saliencyevaluator import GradCamGradientEvaluator  # pylint: disable=import-outside-toplevel
        return GradCamGradientEvaluator(learner, output_dir)
    elif evaluator_id == "deep-lift":
        from seqgra.evaluator.saliencyevaluator import DeepLiftEvaluator  # pylint: disable=import-outside-toplevel
        return DeepLiftEvaluator(learner, output_dir)
    elif evaluator_id == "excitation-backprop":
        from seqgra.evaluator.saliencyevaluator import ExcitationBackpropEvaluator  # pylint: disable=import-outside-toplevel
        return ExcitationBackpropEvaluator(learner, output_dir)
    elif evaluator_id == "contrastive-excitation-backprop":
        from seqgra.evaluator.saliencyevaluator import ContrastiveExcitationBackpropEvaluator  # pylint: disable=import-outside-toplevel
        return ContrastiveExcitationBackpropEvaluator(learner, output_dir)
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


def run_seqgra(data_config_file: str,
               data_folder: str,
               model_config_file: str,
               evaluator_ids: List[str],
               output_dir: str) -> None:
    output_dir = format_output_dir(output_dir.strip())

    if data_config_file is None:
        data_parser_type = None
        simulator_id = data_folder.strip()
        logging.info("loading experimental data")
    else:
        # generate synthetic data
        data_config_file = data_config_file.strip()
        data_config = parse_config_file(data_config_file)
        data_parser: DataParser = XMLDataParser(data_config)
        data_parser_type = data_parser.get_type()
        simulator = Simulator(data_parser, output_dir + "input")
        simulator_id = simulator.id
        print(simulator)
        synthetic_data_available: bool = \
            len(os.listdir(simulator.output_dir)) > 0
        if synthetic_data_available:
            logging.info("loading previously generated synthetic data")
        else:
            logging.info("generating synthetic data")
            simulator.simulate_data()

    # get learner
    if model_config_file is not None:
        model_config = parse_config_file(model_config_file.strip())
        model_parser: ModelParser = XMLModelParser(model_config)
        learner: Learner = get_learner(model_parser, data_parser_type,
                                       output_dir + "input/" + simulator_id,
                                       output_dir + "models/" + simulator_id)

        # load data
        training_set_file: str = learner.get_examples_file("training")
        validation_set_file: str = learner.get_examples_file("validation")
        x_train, y_train = learner.parse_data(training_set_file)
        x_val, y_val = learner.parse_data(validation_set_file)

        # train model on data
        trained_model_available: bool = len(os.listdir(learner.output_dir)) > 0
        if trained_model_available:
            logging.info("loading previously trained model")
            learner.load_model()
        else:
            logging.info("training model")

            learner.create_model()
            learner.print_model_summary()
            learner.train_model(x_train=x_train, y_train=y_train,
                                x_val=x_val, y_val=y_val)
            learner.save_model()

        if evaluator_ids is not None and len(evaluator_ids) > 0:
            evaluation_dir: str = output_dir + "evaluation/" + \
                simulator_id + "/" + learner.id

            evaluators: List[Evaluator] = [get_evaluator(evaluator_id,
                                                         learner,
                                                         evaluation_dir)
                                           for evaluator_id in evaluator_ids]

            logging.info("evaluating model using interpretability methods")
            for evaluator in evaluators:
                logging.info("running evaluator " + evaluator.id)
                evaluator.evaluate_model("training")
                evaluator.evaluate_model("validation")
                evaluator.evaluate_model("test")
        else:
            logging.info("skipping evaluation step: no evaluator specified")


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
        help="evaluator ID or IDs of interpretability method - valid "
        "evaluator IDs include metrics, roc, pr, sis, gradient, "
        "gradientx-input, saliency, integrated-gradient, "
        "nonlinear-integrated-gradient, grad-cam-gradient, deep-lift, "
        "excitation-backprop, contrastive-excitation-backprop"
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        type=str,
        required=True,
        help="output directory, subdirectories are created for generated "
        "data, trained model, and model evaluation"
    )
    args = parser.parse_args()

    if args.datafolder and args.modelconfigfile is None:
        parser.error("-f/--datafolder requires -m/--modelconfigfile.")

    if args.evaluators and args.modelconfigfile is None:
        parser.error("-e/--evaluators requires -m/--modelconfigfile.")

    if args.evaluators is not None:
        for evaluator in args.evaluators:
            if evaluator not in frozenset(["metrics", "predict", "roc",
                                           "pr", "sis"]):
                raise ValueError(
                    "invalid evaluator ID {s!r}".format(s=evaluator))

    run_seqgra(args.dataconfigfile,
               args.datafolder,
               args.modelconfigfile,
               args.evaluators,
               args.outputdir)


if __name__ == "__main__":
    main()
