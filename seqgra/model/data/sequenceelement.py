"""
MIT - CSAIL - Gifford Lab - seqgra

SequenceElement class definition, markup language agnostic

@author: Konstantin Krismer
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class SequenceElement(ABC):
    def __init__(self, sid: str) -> None:
        self.sid: str = sid

    @abstractmethod
    def generate(self) -> str:
        pass

    @abstractmethod
    def get_max_length(self) -> int:
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
                 positions: List[List[Tuple[str, float]]]) -> None:
        super().__init__(sid)
        self.positions: List[List[Tuple[str, float]]] = positions

    def __str__(self):
        str_rep = ["Sequence element (matrix-based):\n",
                   "\tID: ", self.sid, "\n",
                   "\tPPM:\n"]
        for pos in self.positions:
            str_rep += ["\t\t", str(pos), "\n"]
        return ''.join(str_rep)

    def __generate_letter(self, position: List[Tuple[str, float]]) -> str:
        letters = [letter[0] for letter in position]
        p = [letter[1] for letter in position]
        p = [prop / sum(p) for prop in p]
        return np.random.choice(letters, p=p)

    def generate(self) -> str:
        return "".join([self.__generate_letter(position)
                        for position in self.positions])

    def get_max_length(self) -> int:
        return len(self.positions)


class KmerBasedSequenceElement(SequenceElement):
    def __init__(self, sid: str, kmers: List[Tuple[str, float]]) -> None:
        super().__init__(sid)
        self.kmers: List[Tuple[str, float]] = kmers
        self._kmers = [kmer[0] for kmer in self.kmers]
        self._p = [kmer[1] for kmer in self.kmers]
        self._p = [prop / sum(self._p) for prop in self._p]

    def __str__(self):
        str_rep = ["Sequence element (k-mer-based):\n",
                   "\tID: ", self.sid, "\n",
                   "\tk-mers:\n"]
        str_rep += ["\t\t" + kmer[0] + ": " + str(round(kmer[1], 3)) + "\n"
                    for kmer in self.kmers]
        return "".join(str_rep)

    def generate(self) -> str:
        return np.random.choice(self._kmers, p=self._p)

    def get_max_length(self) -> int:
        longest_length: int = 0
        for kmer in self.kmers:
            if len(kmer[0]) > longest_length:
                longest_length = len(kmer[0])
        return longest_length
