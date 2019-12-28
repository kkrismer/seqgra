"""
MIT - CSAIL - Gifford Lab - seqgra

Operation class definition, markup language agnostic

@author: Konstantin Krismer
"""
from typing import Dict

class Operation:
    def __init__(self, name: str, parameters: Dict[str, str]) -> None:
        self.name: str = name
        self.parameters: Dict[str, str] = parameters

    def __str__(self):
        str_rep = ["Operation:\n",
            "\tName:", self.name, "\n",
            "\tParameters:\n",
            "\t\t", self.parameters]
        return "".join(str_rep)
