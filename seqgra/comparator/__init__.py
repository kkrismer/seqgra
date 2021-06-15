"""
seqgra comparators used by `seqgras` command

Classes:
    - :class:`~seqgra.comparator.comparator.Comparator`: abstract base class for all comparators
    - :class:`~seqgra.comparator.curvetablecomparator.CurveTableComparator`: collects PR and ROC curve information in text file
    - :class:`~seqgra.comparator.fietablecomparator.FIETableComparator`: collects feature importance evaluator information in text file
    - :class:`~seqgra.comparator.prcomparator.PRComparator`: creates PR curves from various grammars and architectures
    - :class:`~seqgra.comparator.roccomparator.ROCComparator`: creates ROC curves from various grammars and architectures
    - :class:`~seqgra.comparator.tablecomparator.TableComparator`: collects grammar and model information in text file
"""
from seqgra.comparator.comparator import Comparator
from seqgra.comparator.roccomparator import ROCComparator
from seqgra.comparator.prcomparator import PRComparator
from seqgra.comparator.tablecomparator import TableComparator
from seqgra.comparator.curvetablecomparator import CurveTableComparator
from seqgra.comparator.fietablecomparator import FIETableComparator
