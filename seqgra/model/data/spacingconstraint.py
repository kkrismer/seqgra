"""
MIT - CSAIL - Gifford Lab - seqgra

SpacingConstraint class definition, markup language agnostic

@author: Konstantin Krismer
"""

from seqgra.model.data.sequenceelement import SequenceElement

class SpacingConstraint:
    def __init__(self, sequence_element1: SequenceElement,
                 sequence_element2: SequenceElement, min_distance: int,
                 max_distance: int, direction: str) -> None:
        self.sequence_element1: SequenceElement = sequence_element1
        self.sequence_element2: SequenceElement = sequence_element2
        self.min_distance: int = min_distance
        self.max_distance: int = max_distance
        self.direction: str = direction

    def __str__(self):
        str_rep = ["Spacing constraint:\n",
        "\tFirst sequence element: ", self.sequence_element1.id, " [sid]\n",
        "\tSecond sequence element: ", self.sequence_element2.id, " [sid]\n",
        "\tMinimum distance: ", str(self.min_distance), "\n",
        "\tMaximum distance: ", str(self.max_distance), "\n",
        "\tDirection: ", self.direction, "\n"]
        return "".join(str_rep)