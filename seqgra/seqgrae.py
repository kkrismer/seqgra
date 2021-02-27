#!/usr/bin/env python

"""MIT - CSAIL - Gifford Lab - seqgra

seqgra ensemble

@author: Konstantin Krismer
"""

import argparse
import copy
import logging
import math
import os
import random
import shutil
from typing import List

import pandas as pd

import seqgra
import seqgra.constants as c
from seqgra import MiscHelper
from seqgra.model import DataDefinition
from seqgra.model import ModelDefinition
from seqgra.parser import DataDefinitionParser
from seqgra.parser import XMLDataDefinitionParser
from seqgra.parser import ModelDefinitionParser
from seqgra.parser import XMLModelDefinitionParser
from seqgra.writer import XMLDataDefinitionWriter
from seqgra.writer import XMLModelDefinitionWriter


def parse_data_definition_file(file_name: str) -> DataDefinition:
    data_config = MiscHelper.read_config_file(file_name)
    data_def_parser: DataDefinitionParser = XMLDataDefinitionParser(
        data_config)
    return data_def_parser.get_data_definition()


def parse_model_definition_file(file_name: str) -> ModelDefinition:
    model_config = MiscHelper.read_config_file(file_name)
    model_def_parser: ModelDefinitionParser = XMLModelDefinitionParser(
        model_config)
    return model_def_parser.get_model_definition()


def change_ds_size(data_definition: DataDefinition, ds_size: int,
                   training_set_fraction: float = 0.7,
                   validation_set_fraction: float = 0.1,
                   test_set_fraction: float = 0.2) -> DataDefinition:
    org_training_set_size: int = 0
    org_validation_set_size: int = 0
    org_test_set_size: int = 0
    new_training_set_size: int = (ds_size * training_set_fraction)
    new_validation_set_size: int = (ds_size * validation_set_fraction)
    new_test_set_size: int = (ds_size * test_set_fraction)

    # calculate total number of examples per set
    for example_set in data_definition.data_generation.sets:
        if example_set.name == c.DataSet.TRAINING:
            org_training_set_size = sum([example.samples
                                         for example in example_set.examples])
        elif example_set.name == c.DataSet.VALIDATION:
            org_validation_set_size = sum([example.samples
                                           for example in example_set.examples])
        elif example_set.name == c.DataSet.TEST:
            org_test_set_size = sum([example.samples
                                     for example in example_set.examples])
        else:
            raise Exception("invalid set name: " + example_set.name)

    # set new number of examples proportional to previous example set
    for example_set in data_definition.data_generation.sets:
        if example_set.name == c.DataSet.TRAINING:
            for example in example_set.examples:
                example.samples = int(
                    (example.samples / org_training_set_size) * new_training_set_size)
        elif example_set.name == c.DataSet.VALIDATION:
            for example in example_set.examples:
                example.samples = int(
                    (example.samples / org_validation_set_size) * new_validation_set_size)
        elif example_set.name == c.DataSet.TEST:
            for example in example_set.examples:
                example.samples = int(
                    (example.samples / org_test_set_size) * new_test_set_size)

    return data_definition


def get_grammar_id(org_grammar_id: str, ds_size: int,
                   seed: int) -> str:
    if ds_size % 1000 == 0:
        return org_grammar_id + "-" + \
            str(int(ds_size / 1000)) + "k-s" + str(seed)
    else:
        return org_grammar_id + "-" + \
            str(ds_size) + "-s" + str(seed)


def get_data_folder_name(data_folder, subsampling_rate: float,
                         seed: int) -> str:
    org_data_folder_name: str = os.path.basename(os.path.normpath(data_folder))
    return org_data_folder_name + "-" + \
        str(subsampling_rate) + "-s" + str(seed)


def get_model_id(org_model_id: str, seed: int) -> str:
    return org_model_id + "-s" + str(seed)


def write_data_definition_file(data_definition: DataDefinition,
                               output_dir: str,
                               ds_size: int, seed: int) -> None:
    data_definition = copy.deepcopy(data_definition)
    data_definition = change_ds_size(data_definition, ds_size)
    data_definition.seed = seed
    data_definition.grammar_id = get_grammar_id(data_definition.grammar_id,
                                                ds_size, seed)

    XMLDataDefinitionWriter.write_data_definition_to_file(
        data_definition, output_dir + data_definition.grammar_id + ".xml")


def write_data_definition_files(data_config_file: str,
                                output_dir: str,
                                ds_sizes: List[float],
                                d_seeds: List[int]) -> None:
    data_definition: DataDefinition = parse_data_definition_file(
        data_config_file)

    for ds_size in ds_sizes:
        for d_seed in d_seeds:
            write_data_definition_file(data_definition, output_dir,
                                       ds_size, d_seed)


def subsample_data_set(data_folder, output_dir, new_data_folder,
                       subsampling_rate) -> None:
    out_folder: str = MiscHelper.prepare_path(output_dir + new_data_folder)

    # subsample training set
    example_in_file: str = os.path.normpath(
        data_folder + "/" + c.DataSet.TRAINING + ".txt")
    annotation_in_file: str = os.path.normpath(
        data_folder + "/" + c.DataSet.TRAINING + "-annotation.txt")
    example_out_file: str = os.path.normpath(
        out_folder + "/" + c.DataSet.TRAINING + ".txt")
    annotation_out_file: str = os.path.normpath(
        out_folder + "/" + c.DataSet.TRAINING + "-annotation.txt")

    if math.isclose(subsampling_rate, 1.0):
        shutil.copyfile(example_in_file, example_out_file)
        shutil.copyfile(annotation_in_file, annotation_out_file)
    else:
        example_df = pd.read_csv(example_in_file, sep="\t",
                                 dtype={"x": "string", "y": "string"})
        annotation_df = pd.read_csv(example_in_file, sep="\t",
                                    dtype={"annotation": "string", "y": "string"})

        num_examples: int = len(example_df)
        subsampled_num_examples: int = int(num_examples * subsampling_rate)

        idx = random.sample(range(num_examples), subsampled_num_examples)

        subsampled_example_df = example_df.iloc[idx]
        subsampled_annotation_df = annotation_df.iloc[idx]

        subsampled_example_df.to_csv(example_out_file, sep="\t", index=False)
        subsampled_annotation_df.to_csv(
            annotation_out_file, sep="\t", index=False)

    # copy validation set
    example_in_file: str = os.path.normpath(
        data_folder + "/" + c.DataSet.VALIDATION + ".txt")
    annotation_in_file: str = os.path.normpath(
        data_folder + "/" + c.DataSet.VALIDATION + "-annotation.txt")
    example_out_file: str = os.path.normpath(
        out_folder + "/" + c.DataSet.VALIDATION + ".txt")
    annotation_out_file: str = os.path.normpath(
        out_folder + "/" + c.DataSet.VALIDATION + "-annotation.txt")
    shutil.copyfile(example_in_file, example_out_file)
    shutil.copyfile(annotation_in_file, annotation_out_file)

    # copy test set
    example_in_file: str = os.path.normpath(
        data_folder + "/" + c.DataSet.TEST + ".txt")
    annotation_in_file: str = os.path.normpath(
        data_folder + "/" + c.DataSet.TEST + "-annotation.txt")
    example_out_file: str = os.path.normpath(
        out_folder + "/" + c.DataSet.TEST + ".txt")
    annotation_out_file: str = os.path.normpath(
        out_folder + "/" + c.DataSet.TEST + "-annotation.txt")
    shutil.copyfile(example_in_file, example_out_file)
    shutil.copyfile(annotation_in_file, annotation_out_file)


def subsample_experimental_data(data_folder: str,
                                output_dir: str,
                                subsampling_rates: List[float],
                                d_seeds: List[int]) -> None:
    for subsampling_rate in subsampling_rates:
        for d_seed in d_seeds:
            new_data_folder: str = get_data_folder_name(
                data_folder, subsampling_rate, d_seed)
            subsample_data_set(
                data_folder, output_dir, new_data_folder, subsampling_rate)


def write_model_definition_file(model_definition: ModelDefinition,
                                output_dir: str,
                                seed: int) -> None:
    model_definition = copy.deepcopy(model_definition)
    model_definition.seed = seed
    model_definition.model_id = get_model_id(model_definition.model_id, seed)

    XMLModelDefinitionWriter.write_model_definition_to_file(
        model_definition, output_dir + model_definition.model_id + ".xml")


def write_model_definition_files(model_config_files: List[str],
                                 output_dir: str,
                                 m_seeds: List[int]) -> None:
    for model_config_file in model_config_files:
        model_definition: ModelDefinition = parse_model_definition_file(
            model_config_file)

        for m_seed in m_seeds:
            write_model_definition_file(model_definition, output_dir,
                                        m_seed)


def write_analysis_script(analysis_name: str,
                          data_config_file: str,
                          data_folder: str,
                          model_config_files: List[str],
                          output_dir: str,
                          ds_sizes: List[float],
                          d_seeds: List[int],
                          m_seeds: List[int],
                          seed_grid: bool,
                          gpu: bool) -> None:
    analyses_dir: str = MiscHelper.prepare_path(
        output_dir + "analyses", allow_exists=True, allow_non_empty=True)


def run_seqgra_ensemble(analysis_name: str,
                        data_config_file: str,
                        data_folder: str,
                        model_config_files: List[str],
                        output_dir: str,
                        ds_sizes: List[float],
                        d_seeds: List[int],
                        m_seeds: List[int],
                        seed_grid: bool,
                        gpu: bool) -> None:
    output_dir = MiscHelper.format_output_dir(output_dir.strip())
    model_configs_dir: str = MiscHelper.prepare_path(
        output_dir + "configs/model", allow_exists=True, allow_non_empty=True)

    if data_config_file:
        data_configs_dir: str = MiscHelper.prepare_path(
            output_dir + "configs/data", allow_exists=True,
            allow_non_empty=True)
        write_data_definition_files(data_config_file, data_configs_dir,
                                    ds_sizes, d_seeds)
    else:
        input_data_dir: str = MiscHelper.prepare_path(
            output_dir + "input", allow_exists=True,
            allow_non_empty=True)
        subsample_experimental_data(data_folder, input_data_dir,
                                    ds_sizes, d_seeds)

    write_model_definition_files(model_config_files, model_configs_dir,
                                 m_seeds)

    write_analysis_script(analysis_name, data_config_file, data_folder,
                          model_config_files, output_dir, ds_sizes, d_seeds,
                          m_seeds, seed_grid, gpu)


def main():
    logging.basicConfig(level=logging.INFO)

    default_ds_sizes: List[int] = [10000, 20000, 40000, 80000, 160000,
                                   320000, 640000, 1280000]
    default_subsampling_rates: List[float] = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]

    parser = argparse.ArgumentParser(
        prog="seqgras",
        description="seqgra ensemble: Test model architecture on grammar "
        "across data set sizes, simulation and model seeds")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s " + seqgra.__version__)
    parser.add_argument(
        "-a",
        "--analysis-name",
        type=str,
        required=True,
        help="analysis name (used for file name and comparators)"
    )
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
        "--model-config-files",
        type=str,
        required=True,
        nargs="+",
        help="list of paths to the seqgra XML model configuration files"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="output directory, subdirectories are created for generated "
        "data and model configs, input data, trained models, and model "
        "evaluations"
    )
    parser.add_argument(
        "--ds-sizes",
        type=int,
        default=default_ds_sizes,
        nargs="+",
        help="if -d is specified: list of data set sizes in number of "
        "examples, where train-val-test split is always 70%:10:%20, "
        "defaults to [10000, 20000, 40000, 80000, 160000, 320000, 640000, "
        "1280000]; if -f is specified: list of subsampling rates "
        "of training examples, 1.0 equals original data, no subsampling, "
        "defaults to [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]"
    )
    parser.add_argument(
        "--d-seeds",
        type=int,
        default=[1, 2, 3],
        nargs="+",
        help="list of simulation seeds, defaults to [1, 2, 3]"
    )
    parser.add_argument(
        "--m-seeds",
        type=int,
        default=[1, 2, 3],
        nargs="+",
        help="list of model seeds, defaults to [1, 2, 3]"
    )
    parser.add_argument(
        "--seed-grid",
        action="store_true",
        help="if this flag is set, all simulation and model seed combinations "
        "are evaluated, otherwise simulation seed 1 is only tested with model "
        "seed 1 and so on"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
        help="ID of GPU used by TensorFlow and PyTorch (defaults to GPU "
        "ID 0); CPU is used if no GPU is available or GPU ID is set to -1"
    )
    args = parser.parse_args()

    if args.data_config_file or args.ds_sizes != default_ds_sizes:
        run_seqgra_ensemble(args.analysis_name,
                            args.data_config_file,
                            args.data_folder,
                            args.model_config_files,
                            args.output_dir,
                            args.ds_sizes,
                            args.d_seeds,
                            args.m_seeds,
                            args.seed_grid,
                            args.gpu)
    else:
        run_seqgra_ensemble(args.analysis_name,
                            args.data_config_file,
                            args.data_folder,
                            args.model_config_files,
                            args.output_dir,
                            default_subsampling_rates,
                            args.d_seeds,
                            args.m_seeds,
                            args.seed_grid,
                            args.gpu)


if __name__ == "__main__":
    main()
