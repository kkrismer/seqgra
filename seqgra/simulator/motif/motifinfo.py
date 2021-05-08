"""
MIT - CSAIL - Gifford Lab - seqgra

Motif info

@author: Konstantin Krismer
"""
import math
from typing import List

import pandas as pd

from seqgra import ProbabilisticToken
from seqgra.model import DataDefinition
from seqgra.model.data import AlphabetDistribution
from seqgra.model.data import SequenceElement
from seqgra.model.data import MatrixBasedSequenceElement


class MotifInfo:
    @staticmethod
    def _calculate_position_uncertainty(letter: ProbabilisticToken) -> float:
        return letter.probability * math.log2(letter.probability) * (-1)

    @staticmethod
    def _calculate_information_content(position: List[ProbabilisticToken]) -> float:
        alphabet_size: int = len(position)
        total_information_content: float = math.log2(alphabet_size)
        position_uncertainty: float = 0

        for i in range(alphabet_size):
            position_uncertainty += MotifInfo._calculate_position_uncertainty(position[i])

        return total_information_content - position_uncertainty

    @staticmethod
    def _calculate_motif_information_content(se: MatrixBasedSequenceElement) -> float:
        information_content: float = 0

        for position in se.positions:
            information_content += MotifInfo._calculate_information_content(position)

        return information_content

    @staticmethod
    def _calculate_kl_divergence(position: List[ProbabilisticToken],
                                alphabet_distribution: AlphabetDistribution) -> float:
        alphabet_size: int = len(position)
        kl_divergence: float = 0

        for i in range(alphabet_size):
            if position[i].token != alphabet_distribution.letters[i].token:
                raise Exception("invalid order of token: " +
                                position[i].token + " != " +
                                alphabet_distribution.letters[i].token)
            kl_divergence += position[i].probability * math.log2(
                position[i].probability / alphabet_distribution.letters[i].probability)

        return kl_divergence

    @staticmethod
    def _calculate_motif_kl_divergence(se: MatrixBasedSequenceElement,
                                      alphabet_distribution: AlphabetDistribution) -> float:
        kl_divergence: float = 0

        for position in se.positions:
            kl_divergence += MotifInfo._calculate_kl_divergence(
                position, alphabet_distribution)

        return kl_divergence

    @staticmethod
    def _calculate_motif_info(sequence_elements: List[SequenceElement],
                             alphabet_distribution: AlphabetDistribution) -> pd.DataFrame:
        se_column: List[str] = list()
        width_column: List[int] = list()
        ic_column: List[float] = list()
        kl_divergence_column: List[float] = list()

        for se in sequence_elements:
            if isinstance(se, MatrixBasedSequenceElement):
                se_column += [se.sid]
                width_column += [len(se.positions)]
                ic_column += [MotifInfo._calculate_motif_information_content(se)]
                kl_divergence_column += [MotifInfo._calculate_motif_kl_divergence(
                    se, alphabet_distribution)]

        return pd.DataFrame({"se": se_column,
                             "width": width_column,
                             "information_content": ic_column,
                             "kl_divergence": kl_divergence_column})

    @staticmethod
    def create(output_dir: str, data_definition: DataDefinition) -> None:
        file_name: str = output_dir + "/motif-info.txt"
        df = MotifInfo._calculate_motif_info(
            data_definition.sequence_elements,
            data_definition.background.alphabet_distributions[0])
        df.to_csv(file_name, sep="\t", index=False)
