# seqgra: Principled Selection of Neural Network Architectures for Genomics Prediction Tasks

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Travis build status](https://travis-ci.com/kkrismer/seqgra.svg?branch=master)](https://travis-ci.com/kkrismer/seqgra)

*PyPI version badge placeholder*

*DOI badge placeholder*

https://kkrismer.github.io/seqgra/

## What is seqgra?

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

## Installation

seqgra is a Python package that is part of [PyPI](https://pypi.org/), 
the package repositories behind [pip](https://pip.pypa.io/en/stable/).

To install seqgra with pip, run:
```
pip install seqgra
```

To install seqgra directly from this repository, run:
```
git clone https://github.com/gifford-lab/seqgra
cd seqgra
pip install .
```

### System requirements

- Python 3.7 (or higher)
- *R 3.5 (or higher)*
    - *R package `ggplot2` 3.3.0 (or higher)*
    - *R package `gridExtra` 2.3 (or higher)*
    - *R package `scales` 1.1.0 (or higher)*


The ``tensorflow`` package is only required if TensorFlow models are used 
and will not be automatically installed by ``pip install seqgra``. Same is 
true for packages ``torch`` and ``pytorch-ignite``, which are only 
required if PyTorch models are used.

R is a soft dependency, in the sense that it is used to create a number 
of plots (grammar-model-agreement plots, 
grammar heatmaps, and motif similarity matrix plots) and if not available, 
these plots will be skipped.

seqgra depends upon the Python package [lxml](https://lxml.de/), which in turn 
depends on system libraries that are not always present. On a 
Debian/Ubuntu machine you can satisfy those requirements using:
```
sudo apt-get install libxml2-dev libxslt-dev
```

## Usage

Check out the following help pages:

* [Usage examples](https://kkrismer.github.io/seqgra/examples.html): seqgra example analyses with data definitions and model definitions
* [Command line utilities](https://kkrismer.github.io/seqgra/cmd.html): argument descriptions for `seqgra`, `seqgras`, `seqgrae`, and `seqgraa` commands
* [Data definition](https://kkrismer.github.io/seqgra/dd.html): detailed description of the data definition language that is used to formalize grammars
* [Model definition](https://kkrismer.github.io/seqgra/md.html): detailed description of the model definition language that is used to describe neural network architectures and hyperparameters for the optimizer, the loss, and the training process
* [Simulators, Learners, Evaluators, Comparators](https://kkrismer.github.io/seqgra/slec.html): brief descriptions of the most important classes
* [seqgra API reference](https://kkrismer.github.io/seqgra/seqgra.html): detailed description of the seqgra API
* [Source code](https://github.com/gifford-lab/seqgra): seqgra source code repository on GitHub 

## Citation

If you use seqgra in your work, please cite:

**seqgra: Principled Selection of Neural Network Architectures for Genomics Prediction Tasks**  
Konstantin Krismer, Jennifer Hammelman, and David K. Gifford  
journal name TODO, Volume TODO, Issue TODO, date TODO, Page TODO; DOI: https://doi.org/TODO

## Funding

We gratefully acknowledge funding from NIH grants 1R01HG008754 and 
1R01NS109217.
