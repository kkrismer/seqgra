Introduction
============

What is seqgra?
---------------
Sequence models based on deep neural networks have achieved state-of-the-art 
performance on regulatory genomics prediction tasks, such as chromatin 
accessibility and transcription factor binding. But despite their high 
accuracy, their contributions to a mechanistic understanding of the biology 
of regulatory elements is often hindered by the complexity of the predictive 
model and thus poor interpretability of its decision boundaries. To address 
this, we introduce seqgra, a deep learning pipeline that incorporates the 
rule-based simulation of biological sequence data and the training and 
evaluation of models, whose decision boundaries mirror the rules from the 
simulation process. The method can be used to (1) generate data under the 
assumption of a hypothesized model of genome regulation, (2) identify neural 
network architectures capable of recovering the rules of said model, and (3) 
analyze a model's predictive performance as a function of training set size, 
noise level, and the complexity of the rules behind the simulated data.

Installation
------------
seqgra is a Python package that is part of both conda-forge_ and PyPI_, the 
package repositories behind conda_ and pip_, respectively.

To install seqgra with conda, run:

.. code-block:: shell

    conda install -c conda-forge seqgra

To install seqgra with pip, run:

.. code-block:: shell

    pip install seqgra

To install seqgra directly from this repository, run:

.. code-block:: shell

    git clone https://github.com/gifford-lab/seqgra
    cd seqgra
    pip install .

System requirements
^^^^^^^^^^^^^^^^^^^

- Python 3.7 (or higher)
- R 3.5 (or higher)

  - R package ggplot2 3.3.0 (or higher)
  - R package gridExtra 2.3 (or higher)
  - R package scales 1.1.0 (or higher)

.. note::
    The ``tensorflow`` package is only required if TensorFlow models are used 
    and will not be automatically installed by ``pip install seqgra``. Same is 
    true for packages ``torch`` and ``pytorch-ignite``, which are only 
    required if PyTorch models are used.

.. note::
    R is a soft dependency, in the sense that it is used to create a number 
    of plots (grammar-model-agreement plots, 
    grammar heatmaps, and motif similarity matrix plots) and if not available, 
    these plots will be skipped.

.. note::
    seqgra depends upon the Python package lxml_, which in turn 
    depends on system libraries that are not always present. On a 
    Debian/Ubuntu machine you can satisfy those requirements using:
    
    .. code-block:: shell

        sudo apt-get install libxml2-dev libxslt-dev

Usage
-----
Check out the following help pages:

* :doc:`examples`: seqgra example analyses with data definitions and model 
  definitions
* :doc:`cmd`: argument descriptions for ``seqgra``, ``seqgras``, ``seqgrae``, 
  and ``seqgraa`` commands
* :doc:`dd`: detailed description of the data definition language that is 
  used to formalize grammars
* :doc:`md`: detailed description of the model definition language that is 
  used to describe neural network architectures and hyperparameters for the 
  optimizer, the loss, and the training process
* :doc:`slec`: brief descriptions of the most important classes
* :doc:`API reference<seqgra>`: detailed description of the seqgra API

Citation
--------
If you use seqgra in your work, please cite:

| **Identifying Neural Network Architectures for Genomics Prediction Tasks Using Sequence Grammar Based Simulations**
| Konstantin Krismer, Jennifer Hammelman, and David K. Gifford  
| journal name TODO, Volume TODO, Issue TODO, date TODO, Page TODO; DOI: https://doi.org/TODO

Funding
-------
We gratefully acknowledge funding from NIH grants 1R01HG008754 and 
1R01NS109217.

.. _conda-forge: https://anaconda.org/conda-forge
.. _PyPI: https://pypi.org/
.. _conda: https://docs.conda.io/en/latest/
.. _pip: https://pip.pypa.io/en/stable/
.. _lxml: https://lxml.de/
