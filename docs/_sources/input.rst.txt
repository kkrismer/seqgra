Format of input data
====================

This document describes the format of training, validation, and test set files 
for both synthesized and experimental data. There is no difference between 
synthesized and experimental data when it comes to file format or 
location. However, the language referring to their location is slightly 
different, with ``{OUTPUT_DIR}/input/{GRAMMAR ID}`` referring to the 
synthesized data of ``GRAMMAR ID`` and ``{OUTPUT_DIR}/input/{DATA FOLDER}`` 
referring to the experimental data of experiment ``{DATA FOLDER}``.

Required files
--------------

Each valid input data folder contains the following files:

.. code-block:: text

    {OUTPUT_DIR}
    +-- input
        +-- {GRAMMAR ID}|{DATA FOLDER}
            |-- training.txt
            |-- training-annotation.txt
            |-- validation.txt
            |-- validation-annotation.txt
            |-- test.txt
            |-- test-annotation.txt

For synthesized data, these files are generated automatically.


.. note::
    Synthesized input data folders contain additional files not present in
    input folders of experimental data, including a session info file 
    (``session-info.txt``) and two grammar heatmap files 
    (``*-grammar-heatmap.pdf`` and ``*-grammar-heatmap.txt``) for each set
    (training, validation, test).

File format
-----------

seqgra recognizes primary and secondary input files, which are described in 
detail below. Both categories of files have two columns (tab-delimited) and 
start with a header row. Each subsequent row is a training, validation,
or test example. Row *i* in the secondary file corresponds to row *i* in the 
primary file of the same set (e.g., ``training.txt`` and 
``training-annotation.txt``).

Primary input files
^^^^^^^^^^^^^^^^^^^

``training.txt``, ``validation.txt``, and ``test.txt`` are considered primary 
input files, containing the following columns:

- ``x``: DNA or protein sequence (upper-case IUPAC single letter codes, ambiguity codes not allowed)
- ``y``: one class label (for multi-class classification) or zero, one, or more labels (for multi-label classification); labels are delimited by the character ``|``, e.g., ``label1|label2``

**Example 1 - multi-class classification, 8 classes, 150 bp DNA:**

Classes: ``c1`` to ``c8``

.. code-block:: text

    x ⇥ y
    GGGGCGCTTCTGGTAAATTTTTCTAGATGCCTCGAGGGTGCCGAGGAACCGCTAAAGCCAAAGAACTATAAAAAGACACACCAGGCGAGTCGAGCAGATAGCGTAGGGATCTGCTCTGTGGGTGCGACGCGCACGGGGGCTCCTTGATGT ⇥ c1
    TAGATTGCAGCGTTTACGAGAAATACTGAAGGATGGGGCCAATGCACACCGCAAAGCGTTAGCATCCGTGTCTATGATGCAGAACCCCGTAGAGCGAACCTTGTAGGTCAAATGTCGACTTCTATTTGTCTGACATCCGCATAGAATTTT ⇥ c8
    TCACAAAGATAAAGACGTGCAACCCGGCCATAGTTCCGCCACATAGCGTGACGGGCACTATGCTCGCTGGCTGAATTCGTCTGTTTAGCACCTATCCAACTATTCCAGAACACACAAAGTTCAGCTATCAGAGGCGCGCCTTAACTTTTC ⇥ c8
    TAATAGAGTCCCGAACCTAGTGTTTGTTCACACAGCTTGCGTCGCCCAGCCGTCTTACCACGGGGACGTCCCCCCGTACACGCAAGGGTAGGGTAGTGCTACGGAACGTGATTGTGTTTTAACTCATCTAGCAGAAAATGTGGCTCCCTA ⇥ c2
    GAGCCAGCCCTAACGGCTCGGTTTTGCAAACGTTAAGGTTTCCAGGTACGAATACACTCTACCCGCGGCAAAAAGGACCCAGAGGGTATAGCGCAAGTCAAAAACAATAGTCAACGCTTTATCTGTTCGCCCCGCGAAATAAGATGTGCG ⇥ c1
    GCTCAACTCCCTTTCTTGAAATCCCTTACTTAACACAGAGATTCACCAGCCTGTCTTCGCTAAACGGCATTTTTTTAGAGGTCTACGCAATCTGCCGTGAGTGACACTTCCGTACTCTAGCCCCCTGGCGTTCTGATCCCGGCTGCCTCA ⇥ c3
    [...]

**Example 2 - multi-label classification, 2 labels, 150 bp DNA:**

Labels: ``c1``, ``c2``

.. code-block:: text

    x ⇥ y
    AGGAGATTGAAGGAGGGAAGTGATGTCAGGTGAATTACTGCCCCCTGGTGGCCAGACACAGTAATGGGCTCATTTTGCTGTAATAGTCCTTTGAGTCTTTTCTTCCTGCGTTCCCCTGCCATCTTCCACTACCGGCTACAAAGGGGTTAC ⇥ c1
    GACAGAATGTTTAATCCATTCCAGCCATGCATTTGCATCTCGGTACCGTTTCAATTGGTAGAGTTTGCTTTAGATCTTTAACCTCTACAATGGTCACTTTGGTTTTGTCATTGGGTGTTTGTTTAGAAGGGGATGAAGGGGGTGGGGGAA ⇥ c1
    AGCAGAATTTTTAAAATTTGATCTAGCAGGAGCCAGTGGGCTTTCCCTGTGTCTAGTGTGTCCAAACCGGGTGCAGGTCAGGAGTTAGGCATTTGATTCACTGAAAGATGCAAGGCTAGGAGTATGCCCATTATCTACCAGAACGCCCAG ⇥ 
    GAGCAGCAGTGAGAGGTGGTCGCTAGTAGTGCCTTTTTTCAGCTCTTTGTCAGTTACAGTGTCTTCGGTATCAAGCCTTGAGCCTAAGGTGTGCTTAAACGGCATCTTCCCGATGGGCAGGGCACTGAGAGTTCCTACCGGGCGCTGCCA ⇥ 
    TGCCTTGTGTGTTACAATTGCTAATGATGTTAATACTACTTGTGGAGCATACCAGGACTGGGAAAGATTGGTTGTACAAAAATACAAGGAAAACATTCATCATTCATTTTTTTCTCATCATCATACTCCCCAACCCCACAAGAAATCAAG ⇥ c2
    TAGCTGAGAGTCTAGCCCCTTTTACAGGTCAGAATAGGAAACATTTGCCATCTATTGTCTCTAAGGATGGCCACCTAGGAGACTTCATCTACATAATAAGAACCTTAGTGTCCACCACCCCTTATCTTAATCCAGGCATTCCTTTCTGTT ⇥ c1|c2
    [...]
    

Secondary input files (annotation files)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``training-annotation.txt``, ``validation-annotation.txt``, and 
``test-annotation.txt`` are considered secondary 
input files, containing the following columns:

- ``annotation``: this column annotates each position in the sequence in column ``x`` as either part of the grammar (``G``) or part of the background (``_``) 
- ``y``: same as column ``y`` in primary input file


**Example 1 - multi-class classification, 2 classes, 65 bp DNA:**

Classes: ``c1``, ``c2``

In this example the relevant positions are always the 5 central positions.

.. code-block:: text

    annotation ⇥ y
    ______________________________GGGGG______________________________ ⇥ c2
    ______________________________GGGGG______________________________ ⇥ c2
    ______________________________GGGGG______________________________ ⇥ c1
    ______________________________GGGGG______________________________ ⇥ c1
    ______________________________GGGGG______________________________ ⇥ c1
    ______________________________GGGGG______________________________ ⇥ c2
    ______________________________GGGGG______________________________ ⇥ c2
    ______________________________GGGGG______________________________ ⇥ c2
    ______________________________GGGGG______________________________ ⇥ c2
    ______________________________GGGGG______________________________ ⇥ c1
    [...]

**Example 2 - multi-label classification, 2 labels, 65 bp DNA:**

Labels: ``ESC``, ``DE``

Here the relevant positions are variable and the underlying grammar contains
interactions between different parts of the sequence.

.. code-block:: text

    annotation ⇥ y
    _____________________GGGGGGGG____________________________________ ⇥ ESC
    _____GGGGGGGG_______________________________________GGGGGGGG_____ ⇥ ESC
    _________________________________________________________________ ⇥ 
    GGGG_____________GGGGGGGG___________GGGGGGGG_____________________ ⇥ DE|ESC
    _GGGGGGGG__GGGGGGGG______________________________GGGGGGGG________ ⇥ DE|ESC
    GGGGGGGGGGGG__GGGGGGGG___________________GGGGGGGG________________ ⇥ DE|ESC
    _____________________________________GGGGGGGG____________________ ⇥ ESC
    _______________________________________GGGGGGGG_GGGGGGGG_________ ⇥ DE|ESC
    GGG______________________________________________________________ ⇥ DE
    _________________GGGGGGGG__GGGGGGGG______________________________ ⇥ DE
    [...]
