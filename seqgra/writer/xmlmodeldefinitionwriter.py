"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file writer

@author: Konstantin Krismer
"""
from seqgra.model import ModelDefinition
from seqgra.writer import ModelDefinitionWriter

class XMLModelDefinitionWriter(ModelDefinitionWriter):
    @staticmethod
    def write_model_definition_to_file(model_definition: ModelDefinition,
                                       file_name: str):
        pass
