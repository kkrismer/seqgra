"""
MIT - CSAIL - Gifford Lab - seqgra

Grammar heatmap

@author: Konstantin Krismer
"""
import copy
import logging
import math
import subprocess
from typing import List

import pandas as pd
import pkg_resources

from seqgra import MiscHelper
from seqgra import ProbabilisticToken
from seqgra.model import DataDefinition
from seqgra.model.data import SequenceElement
from seqgra.model.data import MatrixBasedSequenceElement


class KLDivergence:
    @staticmethod
    def _calculate_kl_divergence(se1_position: List[ProbabilisticToken],
                                 se2_position: List[ProbabilisticToken]) -> float:
        kl_divergence: float = 0

        for i in range(len(se1_position)):
            se1_letter: ProbabilisticToken = se1_position[i]
            se2_letter: ProbabilisticToken = se2_position[i]
            if se1_letter.token != se2_letter.token:
                raise Exception("invalid order of token: " +
                                se1_letter.token + " != " + se2_letter.token)

            kl_divergence += se1_letter.probability * \
                math.log2(se1_letter.probability / se2_letter.probability)

        return kl_divergence

    @staticmethod
    def _calculate_total_kl_divergence(se1_matrix: List[List[ProbabilisticToken]],
                                       se2_matrix: List[List[ProbabilisticToken]]) -> float:
        total_kl_divergence: float = 0.0

        if (len(se1_matrix) != len(se2_matrix)):
            raise Exception("PWM size mismatch")

        for i in range(len(se1_matrix)):
            total_kl_divergence += KLDivergence._calculate_kl_divergence(
                se1_matrix[i], se2_matrix[i])
        # total_kl_divergence /= len(se1_matrix)
        return total_kl_divergence

    @staticmethod
    def _get_neutral_position(position: List[ProbabilisticToken]) -> List[ProbabilisticToken]:
        neutral_position = copy.deepcopy(position)
        for i in range(len(neutral_position)):
            neutral_position[i] = ProbabilisticToken(
                position[i].token, 1.0 / len(neutral_position))
        return neutral_position

    @staticmethod
    def _min_kl_divergence(se1: MatrixBasedSequenceElement,
                           se2: MatrixBasedSequenceElement,
                           min_relative_overlap: float = 0.7) -> float:
        se1_matrix: List[List[ProbabilisticToken]] = se1.positions
        se2_matrix: List[List[ProbabilisticToken]] = se2.positions
        min_overlap: int = min(int(math.floor(len(se1_matrix) * min_relative_overlap)),
                               int(math.floor(len(se2_matrix) * min_relative_overlap)))
        neutral_position: List[ProbabilisticToken] = KLDivergence._get_neutral_position(
            se1_matrix[0])

        padding_size: int = abs(len(se1_matrix) - len(se2_matrix))
        min_divergence: float = float("inf")
        divergence: float = float("inf")
        if padding_size > 0:
            for i in range(padding_size + 1):
                if len(se1_matrix) < len(se2_matrix):
                    padded_se1_matrix = copy.deepcopy(se1_matrix)
                    padded_se1_matrix = [
                        neutral_position] * i + padded_se1_matrix + [neutral_position] * (padding_size - i)
                    divergence = KLDivergence._calculate_total_kl_divergence(
                        padded_se1_matrix, se2_matrix)
                else:
                    padded_se2_matrix = copy.deepcopy(se2_matrix)
                    padded_se2_matrix = [
                        neutral_position] * i + padded_se2_matrix + [neutral_position] * (padding_size - i)
                    divergence = KLDivergence._calculate_total_kl_divergence(
                        se1_matrix, padded_se2_matrix)

                if divergence < min_divergence:
                    min_divergence = divergence
        else:
            for i in range(min_overlap + 1):
                padded_se1_matrix = copy.deepcopy(se1_matrix)
                padded_se2_matrix = copy.deepcopy(se2_matrix)
                padded_se1_matrix = [neutral_position] * i + \
                    padded_se1_matrix + [neutral_position] * (min_overlap - i)
                padded_se2_matrix = [
                    neutral_position] * (min_overlap - i) + padded_se2_matrix + [neutral_position] * i
                divergence = KLDivergence._calculate_total_kl_divergence(
                    padded_se1_matrix, padded_se2_matrix)

                if divergence < min_divergence:
                    min_divergence = divergence

            divergence = KLDivergence._calculate_total_kl_divergence(
                se1_matrix, se2_matrix)
            if divergence < min_divergence:
                min_divergence = divergence

        return min_divergence

    @staticmethod
    def _calculate_kl_divergence_matrix(
            sequence_elements: List[SequenceElement],
            silent: bool = False) -> pd.DataFrame:
        se1_column: List[str] = list()
        se2_column: List[str] = list()
        kl_divergence_column: List[float] = list()

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
                        kl_divergence_column += [
                            KLDivergence._min_kl_divergence(se1, se2)]

        return pd.DataFrame({"se1": se1_column,
                             "se2": se2_column,
                             "kl_divergence": kl_divergence_column})

    @staticmethod
    def create(output_dir: str, data_definition: DataDefinition,
               silent: bool = False) -> None:
        logger = logging.getLogger(__name__)

        file_name: str = output_dir + "/motif-kld-matrix.txt"
        df = KLDivergence._calculate_kl_divergence_matrix(
            data_definition.sequence_elements, silent)
        df.to_csv(file_name, sep="\t", index=False)

        plot_script: str = pkg_resources.resource_filename(
            "seqgra", "simulator/motif/similarity.R")

        cmd = ["Rscript", "--no-save", "--no-restore", "--quiet",
               plot_script, output_dir, "kld"]

        try:
            subprocess.call(cmd, universal_newlines=True)
        except subprocess.CalledProcessError as exception:
            logger.warning("failed to create KL divergence matrix: %s",
                           exception.output)
        except FileNotFoundError as exception:
            logger.warning("Rscript not on PATH, skipping "
                           "KL divergence matrix")
