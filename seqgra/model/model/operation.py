"""
MIT - CSAIL - Gifford Lab - seqgra

Operation class definition, markup language agnostic

@author: Konstantin Krismer
"""
from typing import Dict, Optional


class Operation:
    def __init__(self, name: str,
                 parameters: Optional[Dict[str, str]] = None) -> None:
        self.name: str = name
        self.parameters: Optional[Dict[str, str]] = parameters

    def __str__(self):
        str_rep = ["Operation:\n",
                   "\tName:", self.name, "\n",
                   "\tParameters:\n",
                   "\t\t", str(self.parameters), "\n"]
        return "".join(str_rep)
