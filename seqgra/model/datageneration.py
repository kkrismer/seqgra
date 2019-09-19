"""TODO

@author: Konstantin Krismer
"""

from typing import List, Tuple

from seqgra.model.condition import Condition

class DataGeneration:
    def __init__(self, seed: int, conditions: List[Tuple[Condition, int]]) -> None:
        self.seed: int = int(seed)
        self.conditions: List[Tuple[Condition, int]] = conditions
