"""
MIT - CSAIL - Gifford Lab - seqgra

Grammar heatmap

@author: Konstantin Krismer
"""
import logging
import random
from statistics import mean
import subprocess
from typing import List

import numpy as np
import pandas as pd
import pkg_resources

from seqgra import MiscHelper
from seqgra import ProbabilisticToken
from seqgra.learner import DNAHelper
from seqgra.learner.bayes import BayesOptimalHelper
from seqgra.model import DataDefinition
from seqgra.model.data import AlphabetDistribution
from seqgra.model.data import SequenceElement
from seqgra.model.data import MatrixBasedSequenceElement


class EmpiricalSimilarityScore:
    @staticmethod
    def _ess(se: MatrixBasedSequenceElement,
             example: str) -> float:
        pwm = BayesOptimalHelper.se_to_pwm(se)
        encoded_example = DNAHelper.convert_dense_to_one_hot_encoding(example)
        return max(BayesOptimalHelper.score_example(encoded_example, pwm))

    @staticmethod
    def _mean_ess(se1: MatrixBasedSequenceElement,
                  se2: MatrixBasedSequenceElement,
                  padding_size: int = 100,
                  num_examples: int = 100) -> float:
        letters: List[ProbabilisticToken] = [ProbabilisticToken("A", 0.25),
                                             ProbabilisticToken("C", 0.25),
                                             ProbabilisticToken("G", 0.25),
                                             ProbabilisticToken("T", 0.25)]
        alphabet_distribution: AlphabetDistribution = AlphabetDistribution(
            letters)

        similarity_scores: List[float] = list()
        for i in range(num_examples):
            example: str = alphabet_distribution.generate_letters(padding_size)
            example += se2.generate()
            example += alphabet_distribution.generate_letters(padding_size)
            similarity_scores += [EmpiricalSimilarityScore._ess(se1, example)]

        return mean(similarity_scores)

    @staticmethod
    def _calculate_ess_matrix(
            sequence_elements: List[SequenceElement],
            padding_size: int = 100,
            num_examples: int = 100, silent: bool = False) -> pd.DataFrame:
        se1_column: List[str] = list()
        se2_column: List[str] = list()
        empirical_similarity_column: List[float] = list()

        for i, se1 in enumerate(sequence_elements):
            if not silent:
                MiscHelper.print_progress_bar(
                    i + 1, len(sequence_elements),
                    "processing sequence elements:",
                    "- current sequence element: " + se1.sid, 0)
            if isinstance(se1, MatrixBasedSequenceElement):
                for se2 in sequence_elements:
                    if isinstance(se2, MatrixBasedSequenceElement):
                        se1_column += [se1.sid]
                        se2_column += [se2.sid]
                        empirical_similarity_column += [
                            EmpiricalSimilarityScore._mean_ess(
                                se1, se2, padding_size, num_examples)]

        return pd.DataFrame({"se1": se1_column,
                             "se2": se2_column,
                             "empirical_similarity_score": empirical_similarity_column})

    @staticmethod
    def create(output_dir: str, data_definition: DataDefinition,
               padding_size: int = 100, num_examples: int = 100,
               silent: bool = False) -> None:
        logger = logging.getLogger(__name__)

        random.seed(data_definition.seed)
        np.random.seed(data_definition.seed)

        file_name: str = output_dir + "/motif-ess-matrix.txt"
        df = EmpiricalSimilarityScore._calculate_ess_matrix(
            data_definition.sequence_elements, padding_size, num_examples,
            silent)
        df.to_csv(file_name, sep="\t", index=False)

        plot_script: str = pkg_resources.resource_filename(
            "seqgra", "simulator/motif/simmatrix.R")

        cmd = ["Rscript", "--no-save", "--no-restore", "--quiet",
               plot_script, output_dir, "ess"]

        try:
            subprocess.call(cmd, universal_newlines=True)
        except subprocess.CalledProcessError as exception:
            logger.warning("failed to create ESS matrix: %s",
                           exception.output)
        except FileNotFoundError as exception:
            logger.warning("Rscript not on PATH, skipping ESS matrix")
