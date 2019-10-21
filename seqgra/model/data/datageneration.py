"""
MIT - CSAIL - Gifford Lab - seqgra

DataGeneration and Set class definitions, markup language agnostic

@author: Konstantin Krismer
"""
from typing import List, Tuple

from seqgra.model.data.condition import Condition

class Example:
    def __init__(self, samples: int, conditions: List[Condition]) -> None:
        self.samples: int = samples
        self.conditions: List[Condition] = conditions
    
    def __str__(self):
        str_rep: List[str] = ["Example:\n",
        "\tNumber of samples drawn: ", str(self.samples), "\n",
        "\tInstance of the following conditions:\n"]
        conditions_string: List[str] = ["\t\t" + "condition " + condition.id + " [cid]\n" for condition in self.conditions]
        str_rep += conditions_string
        return "".join(str_rep)
    
class ExampleSet:
    def __init__(self, name: str, examples: List[Example]) -> None:
        self.name: str = name
        self.examples: List[Example] = examples
    
    def __str__(self):
        str_rep = ["Set:\n",
        "\tName: ", self.name, "\n",
        "\tExamples:\n"]
        examples_string: List[str] = [str(example) for example in self.examples]
        examples_str_rep = ''.join(examples_string)
        str_rep += ["\t\t" + s + "\n" for s in examples_str_rep.splitlines()]
        return "".join(str_rep)

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
        return "".join(str_rep)

