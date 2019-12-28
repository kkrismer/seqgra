"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file writer

@author: Konstantin Krismer
"""
from typing import List

from seqgra.writer.datawriter import DataWriter
from seqgra.model.data.background import Background
from seqgra.model.data.datageneration import DataGeneration
from seqgra.model.data.condition import Condition
from seqgra.model.data.sequenceelement import SequenceElement

class XMLDataWriter(DataWriter):
    @staticmethod
    def write_data_config(self, file_name: str, id: str, label: str, 
                          description: str, sequence_space: str, type: str, 
                          background: Background, 
                          data_generation: DataGeneration, 
                          conditions: List[Condition], 
                          sequence_elements: List[SequenceElement]):
        pass
