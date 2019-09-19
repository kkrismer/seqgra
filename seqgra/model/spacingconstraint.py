"""TODO

@author: Konstantin Krismer
"""

from seqgra.model.sequenceelement import SequenceElement

class SpacingConstraint:
    def __init__(self, sequence_element1: SequenceElement,
                 sequence_element2: SequenceElement, min_distance: int,
                 max_distance: int, direction: str) -> None:
        self.sequence_element1: str = sequence_element1
        self.sequence_element2: str = sequence_element2
        self.min_distance: int = min_distance
        self.max_distance: int = max_distance
        self.direction: str = direction
