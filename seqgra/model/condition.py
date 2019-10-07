"""MIT - CSAIL - Gifford Lab - seqgra

Condition class definition, markup language agnostic

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
        
    def __str__(self):
        str_rep = ["Condition:\n",
        "\tID: ", self.id, "\n",
        "\tLabel: ", self.label, "\n",
        "\tDescription:\n"]
        if self.description:
            str_rep += ["\t", self.description, "\n"]
        str_rep += ["\tGrammar:\n"]
        rules_string: List[str] = [str(rule) for rule in self.grammar]
        rules_str_rep = ''.join(rules_string)
        str_rep += ["\t\t" + s + "\n" for s in rules_str_rep.splitlines()]
        return "".join(str_rep)

    @staticmethod
    def get_by_id(conditions: List[Condition], id: str) -> Condition:
        for condition in conditions:
            if condition.id == id:
                return condition
        return None