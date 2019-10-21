"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file parser (using Strategy design pattern)

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod
from typing import List

from seqgra.model.background import Background
from seqgra.model.datageneration import DataGeneration
from seqgra.model.condition import Condition
from seqgra.model.sequenceelement import SequenceElement

class DataParser(ABC):
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
    def get_data_generation(self, valid_conditions: List[Condition]) -> DataGeneration:
        pass
    
    @abstractmethod
    def get_conditions(self, valid_sequence_elements: List[SequenceElement]) -> List[Condition]:
        pass
    
    @abstractmethod
    def get_sequence_elements(self) -> List[SequenceElement]:
        pass
