"""
MIT - CSAIL - Gifford Lab - seqgra

SequenceElement class definition, markup language agnostic

@author: Konstantin Krismer
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from seqgra import ProbabilisticToken


class SequenceElement(ABC):
    def __init__(self, sid: str) -> None:
        self.sid: str = sid

    @abstractmethod
    def generate(self) -> str:
        pass

    @abstractmethod
    def get_max_length(self) -> int:
        pass

    @abstractmethod
    def normalize_probabilities(self) -> None:
        pass

    @staticmethod
    def get_by_id(sequence_elements: List[SequenceElement],
                  sid: str) -> SequenceElement:
        for sequence_element in sequence_elements:
            if sequence_element.sid == sid:
                return sequence_element
        return None


class MatrixBasedSequenceElement(SequenceElement):
    def __init__(self, sid: str,
                 positions: List[List[ProbabilisticToken]]) -> None:
        super().__init__(sid)
        self.positions: List[List[ProbabilisticToken]] = positions
        self.normalize_probabilities()

    def __str__(self):
        str_rep = ["Sequence element (matrix-based):\n",
                   "\tID: ", self.sid, "\n",
                   "\tPPM:\n"]
        for pos in self.positions:
            str_rep += ["\t\t", str(pos), "\n"]
        return ''.join(str_rep)

    def normalize_probabilities(self) -> None:
        self.positions = [self.__normalize_position_probabilities(position)
                          for position in self.positions]

    def __normalize_position_probabilities(
            self, position: List[ProbabilisticToken]) -> List[ProbabilisticToken]:
        probabilities = [letter.probability for letter in position]
        probabilities = [p / sum(probabilities) for p in probabilities]
        return [ProbabilisticToken(position[i].token, probabilities[i])
                for i in range(len(position))]

    def generate(self) -> str:
        return "".join([self.__generate_letter(position)
                        for position in self.positions])

    def get_max_length(self) -> int:
        return len(self.positions)

    def __generate_letter(self, position: List[ProbabilisticToken]) -> str:
        letters = [letter.token for letter in position]
        probabilities = [letter.probability for letter in position]
        probabilities = [p / sum(probabilities) for p in probabilities]
        return np.random.choice(letters, p=probabilities)


class KmerBasedSequenceElement(SequenceElement):
    def __init__(self, sid: str, kmers: List[ProbabilisticToken]) -> None:
        super().__init__(sid)
        self.kmers: List[ProbabilisticToken] = kmers
        self._kmers: List[str] = [kmer.token for kmer in self.kmers]
        self._probabilities: List[float] = [kmer.probability
                                            for kmer in self.kmers]
        self.normalize_probabilities()

    def normalize_probabilities(self) -> None:
        self._kmers = [kmer.token for kmer in self.kmers]
        self._probabilities = [kmer.probability
                               for kmer in self.kmers]
        self._probabilities = [p / sum(self._probabilities)
                               for p in self._probabilities]
        self.kmers = [ProbabilisticToken(self._kmers[i],
                                         self._probabilities[i])
                      for i in range(len(self.kmers))]

    def __str__(self):
        str_rep = ["Sequence element (k-mer-based):\n",
                   "\tID: ", self.sid, "\n",
                   "\tk-mers:\n"]
        str_rep += ["\t\t" + kmer.token + ": " +
                    str(round(kmer.probability, 3)) + "\n"
                    for kmer in self.kmers]
        return "".join(str_rep)

    def generate(self) -> str:
        return np.random.choice(self._kmers, p=self._probabilities)

    def get_max_length(self) -> int:
        longest_length: int = 0
        for kmer in self.kmers:
            if len(kmer.token) > longest_length:
                longest_length = len(kmer.token)
        return longest_length
