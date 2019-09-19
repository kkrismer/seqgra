"""TODO

@author: Konstantin Krismer
"""

import logging

from typing import List

from seqgra.parser.parser import Parser
from seqgra.model.background import Background
from seqgra.model.datageneration import DataGeneration
from seqgra.model.condition import Condition
from seqgra.model.sequenceelement import SequenceElement

class Simulator:
    def __init__(self, parser: Parser) -> None:
        self._parser: Parser = parser
        self.__parse_config()

    def __parse_config(self):
        self.id: str = self._parser.get_id()
        self.label: str = self._parser.get_label()
        self.description: str = self._parser.get_description()
        self.sequence_space: str = self._parser.get_sequence_space()
        self.background: Background = self._parser.get_background()
        self.sequence_elements: List[SequenceElement] = self._parser.get_sequence_elements()
        self.conditions: List[Condition] = self._parser.get_conditions(self.sequence_elements)
        self.data_generation: DataGeneration = self._parser.get_data_generation(self.conditions)

    def simulate_data(self, output_dir: str) -> None:
        pass