"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file writer

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod

from seqgra.model import ModelDefinition


class ModelDefinitionWriter(ABC):
    @staticmethod
    @abstractmethod
    def write_model_definition_to_file(model_definition: ModelDefinition,
                                       file_name: str):
        pass
