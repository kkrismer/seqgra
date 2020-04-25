"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file writer

@author: Konstantin Krismer
"""
from seqgra.model import DataDefinition
from seqgra.writer import DataDefinitionWriter


class XMLDataDefinitionWriter(DataDefinitionWriter):
    @staticmethod
    def write_data_definition_to_file(data_definition: DataDefinition,
                                      file_name: str):
        pass
