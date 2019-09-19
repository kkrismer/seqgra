"""TODO

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

    def generate(self) -> str:
        return "TODO"

class KmerBasedSequenceElement(SequenceElement):
    def __init__(self, id: str, kmers: List[Tuple[str, float]]) -> None:
        self.id: str = id
        self.kmers: List[Tuple[str, float]] = kmers

    def generate(self) -> str:
        return "TODO"
