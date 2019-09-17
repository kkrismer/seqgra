"""Gifford Lab - seq-grammar
Abstract base class for parser for configuration files (using Strategy design pattern)

@author: Konstantin Krismer
"""

from typing import List

from abc import ABC, abstractmethod
from seqgra.model.background import Background
from model.datageneration import DataGeneration
from model.condition import Condition
from model.sequenceelement import SequenceElement

class Parser(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass
    
    @abstractmethod
    def get_label(self) -> str:
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        pass
    
    @abstractmethod
    def get_sequence_space(self) -> str:
        pass

    @abstractmethod
    def get_background(self) -> Background:
        pass
    
    @abstractmethod
    def get_data_generation(self) -> DataGeneration:
        pass
    
    @abstractmethod
    def get_conditions(self) -> List[Condition]:
        pass
    
    @abstractmethod
    def get_sequence_elements(self) -> List[SequenceElement]:
        pass
