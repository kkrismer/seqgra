"""TODO

@author: Konstantin Krismer
"""

from abc import ABC, abstractmethod

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

class MatrixBasedSequenceElement(SequenceElement):
    def generate(self) -> str:
        return "TODO"

class KmerBasedSequenceElement(SequenceElement):
    def generate(self) -> str:
        return "TODO"
