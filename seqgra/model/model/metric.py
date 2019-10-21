"""
MIT - CSAIL - Gifford Lab - seqgra

Metric class definition, markup language agnostic

@author: Konstantin Krismer
"""
class Metric:
    def __init__(self, name: str, set_name: str) -> None:
        self.name: str = name
        self.set_name: str = set_name

    def __str__(self):
        str_rep = ["Metric:\n",
            "\tName: ", self.name, "\n",
            "\tSet: ", self.set_name, "\n"]
        return "".join(str_rep)
