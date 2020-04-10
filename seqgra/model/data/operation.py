"""
MIT - CSAIL - Gifford Lab - seqgra

Each instance of PostProcessingOperation has a name
(e.g., "kmer-frequency-preserving-shuffle"), a value used as label (e.g.,
"background" or "class1|class2|class3"), and zero or more parameters (e.g., 
{"k": "3"} for a trimer frequency preserving shuffle).

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List, Dict


class PostprocessingOperation:
    def __init__(self, name: str, labels: str,
                 parameters: Dict[str, str]) -> None:
        self.name: str = name
        self.labels: str = labels
        self.parameters: Dict[str, str] = parameters

    def __str__(self):
        config: List[str] = ["Post-processing operation:\n"]
        config += ["\tname: ", self.name, "\n"]
        config += ["\tlabels: ", self.labels, "\n"]
        config += ["\tparameters:\n"]
        if self.parameters is None or len(self.parameters) == 0:
            config += ["\t\tnone\n"]
        else:
            params: List[str] = [("\t\t" + param_name + ": " + param_value + "\n")
                                 for param_name, param_value in self.parameters.items()]
            config += params
        return "".join(config)
