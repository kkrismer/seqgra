"""
MIT - CSAIL - Gifford Lab - seqgra

Architecture class definition, markup language agnostic

@author: Konstantin Krismer
"""

from typing import Dict, List

from seqgra.model.model.operation import Operation

class Architecture:
    def __init__(self, operations: List[Operation], hyperparameters: Dict[str, str]) -> None:
        self.operations: List[Operation] = operations
        self.hyperparameters: Dict[str, str] = hyperparameters

    def __str__(self):
        str_rep = ["Architecture:\n"]

        if self.operations is not None and len(self.operations) > 0:
            str_rep += ["\tSequential:\n"]
            operators_string: List[str] = [str(operation) for operation in self.operations]
            operators_str_rep = "".join(operators_string)
            str_rep += ["\t\t" + s + "\n" for s in operators_str_rep.splitlines()]

        if self.hyperparameters is not None and len(self.hyperparameters) > 0:
            str_rep += ["\tHyperparameters:\n", "\t\t", str(self.hyperparameters)]

        return "".join(str_rep)
