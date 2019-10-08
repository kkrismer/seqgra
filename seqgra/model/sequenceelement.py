"""
MIT - CSAIL - Gifford Lab - seqgra

SequenceElement class definition, markup language agnostic

@author: Konstantin Krismer
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

class SequenceElement(ABC):
    
    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, id: str) -> None:
        self._id = id

    @abstractmethod
    def generate(self) -> str:
        pass

    @abstractmethod
    def get_max_length(self) -> int:
        pass
    
    @staticmethod
    def get_by_id(sequence_elements: List[SequenceElement], id: str) -> SequenceElement:
        for sequence_element in sequence_elements:
            if sequence_element.id == id:
                return sequence_element
        return None

class MatrixBasedSequenceElement(SequenceElement):
    def __init__(self, id: str, positions: List[List[Tuple[str, float]]]) -> None:
        self.id: str = id
        self.positions: List[List[Tuple[str, float]]] = positions

    def __str__(self):
        str_rep = ["Sequence element (matrix-based):\n",
        "\tID: ", self.id, "\n",
        "\tPPM:\n"]
        for pos in self.positions:
            str_rep += ["\t\t", str(pos), "\n"]
        return ''.join(str_rep)

    def generate(self) -> str:
        return "TODO"
    
    def get_max_length(self) -> int:
        return len(self.positions)

class KmerBasedSequenceElement(SequenceElement):
    def __init__(self, id: str, kmers: List[Tuple[str, float]]) -> None:
        self.id: str = id
        self.kmers: List[Tuple[str, float]] = kmers

    def __str__(self):
        str_rep = ["Sequence element (k-mer-based):\n",
        "\tID: ", self.id, "\n",
        "\tk-mers:\n"]
        str_rep += ["\t\t" + kmer[0] + ": " + str(round(kmer[1], 3)) + "\n" for kmer in self.kmers]
        return ''.join(str_rep)

    def generate(self) -> str:
        return "TODO"
    
    def get_max_length(self) -> int:
        longest_length: int = 0
        for kmer in self.kmers:
            if len(kmer[0]) > longest_length:
                longest_length = len(kmer[0])
        return longest_length
