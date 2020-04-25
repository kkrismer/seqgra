from __future__ import annotations

from typing import List, Optional

from seqgra.model.data import Background
from seqgra.model.data import DataGeneration
from seqgra.model.data import Condition
from seqgra.model.data import SequenceElement


class DataDefinition:
    def __init__(self, grammar_id: str = "", name: str = "", description: str = "",
                 sequence_space: str = "DNA",
                 model_type: str = "multi-class classification",
                 background: Optional[Background] = None,
                 data_generation: Optional[DataGeneration] = None,
                 conditions: Optional[List[Condition]] = None,
                 sequence_elements: Optional[List[SequenceElement]] = None) -> None:
        self.grammar_id: str = grammar_id
        self.name: str = name
        self.description: str = description
        self.sequence_space: str = sequence_space
        self.model_type: str = model_type
        self.background: Optional[Background] = background
        self.data_generation: Optional[DataGeneration] = data_generation
        self.conditions: Optional[List[Condition]] = conditions
        self.sequence_elements: Optional[List[SequenceElement]
                                         ] = sequence_elements

    def __str__(self):
        str_rep: List[str] = ["seqgra data definition:\n",
                              "\tGeneral:\n",
                              "\t\tID: " + self.grammar_id + " [gid]\n",
                              "\t\tName: " + self.name + "\n",
                              "\t\tDescription:\n"]
        if self.description:
            str_rep += ["\t\t\t", self.description, "\n"]
        str_rep += ["\t\tSequence space: " + self.sequence_space + "\n",
                    "\t\tModel type: " + self.model_type + "\n"]
        str_rep += ["\t" + s + "\n"
                    for s in str(self.background).splitlines()]
        str_rep += ["\t" + s + "\n"
                    for s in str(self.data_generation).splitlines()]
        str_rep += ["\tConditions:\n"]
        if self.conditions is not None and len(self.conditions) > 0:
            for condition in self.conditions:
                str_rep += ["\t\t" + s + "\n" for s in str(condition).splitlines()]
        else:
            str_rep += ["\t\tnone"]
        str_rep += ["\tSequence elements:\n"]
        if self.sequence_elements is not None and len(self.sequence_elements) > 0:
            for sequence_element in self.sequence_elements:
                str_rep += ["\t\t" + s +
                            "\n" for s in str(sequence_element).splitlines()]
        else:
            str_rep += ["\t\tnone"]

        return "".join(str_rep)
