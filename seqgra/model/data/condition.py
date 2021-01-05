"""
MIT - CSAIL - Gifford Lab - seqgra

Condition class definition, markup language agnostic

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List

from seqgra.model.data import Rule


class Condition:
    def __init__(self, condition_id: str, label: str,
                 description: str, mode: str, grammar: List[Rule]) -> None:
        self.condition_id: str = condition_id
        self.label: str = label
        self.description: str = description
        self.mode: str = mode
        self.grammar: List[Rule] = grammar

    def __str__(self):
        str_rep = ["Condition:\n",
                   "\tID: ", self.condition_id, "\n",
                   "\tLabel: ", self.label, "\n",
                   "\tDescription:\n"]
        if self.description:
            str_rep += ["\t", self.description, "\n"]
        str_rep += ["\tMode: ", self.mode, "\n"]
        str_rep += ["\tGrammar:\n"]
        rules_string: List[str] = [str(rule) for rule in self.grammar]
        rules_str_rep = "".join(rules_string)
        str_rep += ["\t\t" + s + "\n" for s in rules_str_rep.splitlines()]
        return "".join(str_rep)

    @staticmethod
    def get_by_id(conditions: List[Condition], condition_id: str) -> Condition:
        for condition in conditions:
            if condition.condition_id == condition_id:
                return condition
        return None
