"""TODO

@author: Konstantin Krismer
"""

from typing import List, Tuple

from seqgra.model.sequenceelement import SequenceElement
from seqgra.model.spacingconstraint import SpacingConstraint

class Rule:
    def __init__(self, position: str, sequence_elements: List[Tuple[SequenceElement, float]],
                 spacing_constraints: List[SpacingConstraint]) -> None:
        self.position: str = position
        self.sequence_elements: List[Tuple[SequenceElement, float]] = sequence_elements
        self.spacing_constraints: List[SpacingConstraint] = spacing_constraints
