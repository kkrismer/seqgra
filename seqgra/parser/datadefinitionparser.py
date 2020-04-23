"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file parser
(using Strategy design pattern)

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod
from typing import List

from seqgra.model import DataDefinition
from seqgra.model.data import Background
from seqgra.model.data import DataGeneration
from seqgra.model.data import Condition
from seqgra.model.data import SequenceElement

class DataDefinitionParser(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        pass
    
    @abstractmethod
    def get_sequence_space(self) -> str:
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        pass

    @abstractmethod
    def get_background(self) -> Background:
        pass
    
    @abstractmethod
    def get_data_generation(
        self,
        valid_conditions: List[Condition]) -> DataGeneration:
        pass
    
    @abstractmethod
    def get_conditions(
        self, 
        valid_sequence_elements: List[SequenceElement]) -> List[Condition]:
        pass
    
    @abstractmethod
    def get_sequence_elements(self) -> List[SequenceElement]:
        pass
    
    @abstractmethod
    def get_data_definition(self) -> DataDefinition:
        pass
