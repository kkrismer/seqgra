"""TODO

@author: Konstantin Krismer
"""

from __future__ import annotations

from typing import List, Tuple

class AlphabetDistribution:
    def __init__(self, letters: List[Tuple[str, float]], condition: str = None) -> None:
        self.letters: List[Tuple[str, float]] = letters
        self.condition: str = condition
        self.condition_independent: bool = condition is None
