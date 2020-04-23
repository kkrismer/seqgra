"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file writer

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod

from seqgra.model import DataDefinition


class DataDefinitionWriter(ABC):
    @staticmethod
    @abstractmethod
    def write_data_definition_to_file(data_definition: DataDefinition,
                                      file_name: str):
        pass
