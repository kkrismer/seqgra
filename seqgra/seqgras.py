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
from typing import List, Optional

import seqgra
import seqgra.constants as c
from seqgra import MiscHelper
from seqgra.comparator import Comparator
from seqgra.idresolver import IdResolver


def get_all_grammar_ids(output_dir: str) -> List[str]:
    folder = output_dir + "evaluation/"
    return [o for o in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, o))]


def get_all_model_ids(output_dir: str, grammar_ids: List[str]) -> List[str]:
    model_ids: List[str] = []

    for grammar_id in grammar_ids:
        folder = output_dir + "evaluation/" + grammar_id + "/"
        model_ids += [o for o in os.listdir(folder)
                      if os.path.isdir(os.path.join(folder, o))]
    return list(set(model_ids))


def run_seqgra_summary(analysis_id: str,
                       comparator_ids: List[str],
                       output_dir: str,
                       grammar_ids: Optional[List[str]] = None,
                       model_ids: Optional[List[str]] = None,
                       set_names: Optional[List[str]] = None,
                       model_labels: Optional[List[str]] = None) -> None:
    analysis_id = MiscHelper.sanitize_id(analysis_id)
    output_dir = MiscHelper.format_output_dir(output_dir.strip())

    if comparator_ids:
        for comparator_id in comparator_ids:
            comparator: Comparator = IdResolver.get_comparator(analysis_id,
                                                               comparator_id,
                                                               output_dir,
                                                               model_labels)
            if not grammar_ids:
                grammar_ids = get_all_grammar_ids(output_dir)
            if not model_ids:
                model_ids = get_all_model_ids(output_dir, grammar_ids)

            comparator.compare_models(grammar_ids, model_ids, set_names)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="seqgras",
        description="seqgra summary: Gather metrics across grammars, models, "
        "evaluators")
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
        help="analysis id (folder name for output)"
    )
    parser.add_argument(
        "-c",
        "--comparators",
        type=str,
        required=True,
        nargs="+",
        help="comparator ID or IDs: IDs of "
        "comparators include " +
        ", ".join(sorted(c.ComparatorID.ALL_COMPARATOR_IDS))
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
        "-g",
        "--grammar-ids",
        type=str,
        default=None,
        nargs="+",
        help="one or more grammar IDs; defaults to all grammar IDs in "
        "output dir"
    )
    parser.add_argument(
        "-m",
        "--model-ids",
        type=str,
        default=None,
        nargs="+",
        help="one or more model IDs; defaults to all model IDs for specified "
        "grammars in output dir"
    )
    parser.add_argument(
        "-s",
        "--sets",
        type=str,
        default=["test"],
        nargs="+",
        help="one or more of the following: training, validation, or test"
    )
    parser.add_argument(
        "-l",
        "--model-labels",
        type=str,
        default=None,
        nargs="+",
        help="labels for models, must be same length as model_ids"
    )
    args = parser.parse_args()

    for comparator in args.comparators:
        if comparator not in c.ComparatorID.ALL_COMPARATOR_IDS:
            raise ValueError(
                "invalid comparator ID {s!r}".format(s=comparator))

    run_seqgra_summary(args.analysis_id,
                       args.comparators,
                       args.output_dir,
                       args.grammar_ids,
                       args.model_ids,
                       args.sets,
                       args.model_labels)


if __name__ == "__main__":
    main()
