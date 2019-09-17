"""TODO

@author: Konstantin Krismer
"""

import logging
import xml.dom.minidom
from parser.parser import Parser

class Simulator:
    def __init__(self, parser: Parser) -> None:
        self._parser = parser

    def __parse_config(self):
        self.id = self._parser.get_id()
        self.label = self._parser.get_label()
        self.description = self._parser.get_description()
        self.sequence_space = self._parser.get_sequence_space()
        self.background = self._parser.get_background()
        self.data_generation = self._parser.get_data_generation()
        self.conditions = self._parser.get_conditions()
        self.sequence_elements = self._parser.get_sequence_elements()
