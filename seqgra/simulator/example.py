"""
MIT - CSAIL - Gifford Lab - seqgra

Example class

@author: Konstantin Krismer
"""


class Example:
    def __init__(self, sequence: str, annotation: str) -> None:
        self.sequence: str = sequence
        self.annotation: str = annotation

    def __str__(self):
        str_rep = ["Example:\n",
                   "\tSequence: ", self.sequence, "\n",
                   "\tAnnotation: ", self.annotation, "\n"]
        return "".join(str_rep)
