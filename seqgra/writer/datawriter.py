"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file writer

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod
from typing import List

from seqgra.model.data.background import Background
from seqgra.model.data.datageneration import DataGeneration
from seqgra.model.data.condition import Condition
from seqgra.model.data.sequenceelement import SequenceElement

class DataWriter(ABC):
    @staticmethod
    @abstractmethod
    def write_data_config(self, file_name: str, id: str, label: str, 
                          description: str, sequence_space: str, type: str, 
                          background: Background, 
                          data_generation: DataGeneration, 
                          conditions: List[Condition], 
                          sequence_elements: List[SequenceElement]):
        pass
