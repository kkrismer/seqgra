"""TODO

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List

from seqgra.model.rule import Rule

class Condition:
    def __init__(self, id: str, label: str,
                 description: str, grammar: List[Rule]) -> None:
        self.id: str = id
        self.label: str = label
        self.description: str = description
        self.grammar: List[Rule] = grammar

    @staticmethod
    def get_by_id(conditions: List[Condition], id: str) -> Condition:
        for condition in conditions:
            if condition.id == id:
                return condition
        return None