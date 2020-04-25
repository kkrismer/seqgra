"""
MIT - CSAIL - Gifford Lab - seqgra

Rule class definition, markup language agnostic

@author: Konstantin Krismer
"""
from typing import List, Optional

from seqgra.model.data import SequenceElement
from seqgra.model.data import SpacingConstraint


class Rule:
    def __init__(self, position: str, probability: float,
                 sequence_elements: List[SequenceElement],
                 spacing_constraints: Optional[List[SpacingConstraint]] = None) -> None:
        self.position: str = position
        self.probability: float = probability
        self.sequence_elements: List[SequenceElement] = sequence_elements
        self.spacing_constraints: Optional[List[SpacingConstraint]
                                           ] = spacing_constraints

    def __str__(self):
        str_rep = ["Rule:\n",
                   "\tPosition: ", self.position, "\n",
                   "\tProbability: ", str(self.probability), "\n",
                   "\tSequence elements:\n"]
        str_rep += ["\t\t" + sequence_element.sid + " [sid]\n"
                    for sequence_element in self.sequence_elements]
        if self.spacing_constraints is not None and \
           len(self.spacing_constraints) > 0:
            str_rep += ["\tSpacing constraints:\n"]
            spacing_constraints_string: List[str] = [str(spacing_constraint)
                                                     for spacing_constraint
                                                     in self.spacing_constraints]
            spacing_constraints_str_rep = ''.join(spacing_constraints_string)
            str_rep += ["\t\t" + s + "\n"
                        for s in spacing_constraints_str_rep.splitlines()]
        return "".join(str_rep)
