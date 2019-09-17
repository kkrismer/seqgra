"""TODO

@author: Konstantin Krismer
"""

from __future__ import annotations

from typing import List

class AlphabetDistribution:
    def __init__(self, letters: List[Letter], condition: str = None) -> None:
        self.letters = letters
        self.condition = condition
        self.condition_independent = condition is None

class Letter:
    def __init__(self, letter: str, probability: float) -> None:
        self.letter = letter
        self.probability = float(probability)
