"""MIT - CSAIL - Gifford Lab - seqgra

DataGeneration and Set class definitions, markup language agnostic

@author: Konstantin Krismer
"""

from typing import List, Tuple

from seqgra.model.condition import Condition

class ExampleSet:
    def __init__(self, name: str, conditions: List[Tuple[Condition, int]]) -> None:
        self.name: str = name
        self.conditions: List[Tuple[Condition, int]] = conditions
    
    def __str__(self):
        str_rep = ["Set:\n",
        "\tName: ", self.name, "\n",
        "\tExamples:\n"]
        examples_string: List[str] = ["\t\t" + str(condition[1]) + " examples from condition " + condition[0].id + " [cid]\n" for condition in self.conditions]
        str_rep += examples_string
        return ''.join(str_rep)

class DataGeneration:
    def __init__(self, seed: int, sets: List[ExampleSet]) -> None:
        self.seed: int = int(seed)
        self.sets: List[ExampleSet] = sets
        
    def __str__(self):
        str_rep = ["Data generation:\n",
        "\tSets:\n"]
        sets_string: List[str] = [str(example_set) for example_set in self.sets]
        sets_str_rep = ''.join(sets_string)
        str_rep += ["\t\t" + s + "\n" for s in sets_str_rep.splitlines()]
        return ''.join(str_rep)

