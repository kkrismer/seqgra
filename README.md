# seqgra: Synthetic rule-based biological sequence data generation for architecture evaluation and search

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

seqgra is a Python package that is part of both 
[conda-forge](https://anaconda.org/conda-forge) and [PyPI](https://pypi.org/), 
the package repositories behind [conda](https://docs.conda.io/en/latest/) and 
[pip](https://pip.pypa.io/en/stable/), respectively.

To install seqgra with conda, run:
```
conda install -c conda-forge seqgra
```

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

## Citation

If you use seqgra in your work, please cite:

| **Identifying Neural Network Architectures for Genomics Prediction Tasks Using Sequence Grammar Based Simulations**
| Konstantin Krismer, Jennifer Hammelman, and David K. Gifford  
| journal name TODO, Volume TODO, Issue TODO, date TODO, Page TODO; DOI: https://doi.org/TODO

## Funding

We gratefully acknowledge funding from NIH grants 1R01HG008754 and 
1R01NS109217.


## Examples of seqgra analyses

**Generate synthetic data only:**
```
seqgra -d DATA_CONFIG_FILE \
       -o OUTPUT_DIR
```

Generated files and folders:
<pre>
{OUTPUT_DIR}
+-- input
    +-- {GRAMMAR ID}
        |-- session-info.txt
        |-- training.txt
        |-- training-annotation.txt
        |-- training-grammar-heatmap.txt
        |-- training-grammar-heatmap.pdf
        |-- validation.txt
        |-- validation-annotation.txt
        |-- validation-grammar-heatmap.txt
        |-- validation-grammar-heatmap.pdf
        |-- test.txt
        |-- test-annotation.txt
        |-- test-grammar-heatmap.txt
        +-- test-grammar-heatmap.pdf
</pre>

**Generate synthetic data and train model on it:**
```
seqgra -d DATA_CONFIG_FILE \
       -m MODEL_CONFIG_FILE \
       -o OUTPUT_DIR
```

Generated files and folders:
<pre>
{OUTPUT_DIR}
|-- input
|   +-- {GRAMMAR ID}
|       |-- session-info.txt
|       |-- training.txt
|       |-- training-annotation.txt
|       |-- training-grammar-heatmap.txt
|       |-- training-grammar-heatmap.pdf
|       |-- validation.txt
|       |-- validation-annotation.txt
|       |-- validation-grammar-heatmap.txt
|       |-- validation-grammar-heatmap.pdf
|       |-- test.txt
|       |-- test-annotation.txt
|       |-- test-grammar-heatmap.txt
|       +-- test-grammar-heatmap.pdf
+-- models
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            |-- last-epoch-completed.txt
            |-- num-model-parameters.txt
            |-- saved_model*
            +-- session-info.txt
</pre>

**Train model on previously synthesized data:**
```
seqgra -d DATA_CONFIG_FILE \
       -m MODEL_CONFIG_FILE \
       -o OUTPUT_DIR
```
where previously synthesized data is in the `{OUTPUT_DIR}/input/{GRAMMAR ID}` 
folder and `{GRAMMAR ID}` is defined in `{DATA_CONFIG_FILE}`.

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUT_DIR}</i>
|-- <i>input</i>
|   +-- <i>{GRAMMAR ID}</i>
|       |-- <i>session-info.txt</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>training-grammar-heatmap.txt</i>
|       |-- <i>training-grammar-heatmap.pdf</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>validation-grammar-heatmap.txt</i>
|       |-- <i>validation-grammar-heatmap.pdf</i>
|       |-- <i>test.txt</i>
|       |-- <i>test-annotation.txt</i>
|       |-- <i>test-grammar-heatmap.txt</i>
|       +-- <i>test-grammar-heatmap.pdf</i>
+-- models
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            |-- last-epoch-completed.txt
            |-- num-model-parameters.txt
            |-- saved_model*
            +-- session-info.txt
</pre>

**Train model on experimental or externally synthesized data:**
```
seqgra -f DATA_FOLDER \
       -m MODEL_CONFIG_FILE \
       -o OUTPUT_DIR
```
where experimental or externally synthesized data is in the 
`{OUTPUT_DIR}/input/{DATA_FOLDER}` folder.

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUT_DIR}</i>
|-- <i>input</i>
|   +-- <i>{DATA_FOLDER}</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>test.txt</i>
|       +-- <i>test-annotation.txt</i>
+-- models
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            |-- last-epoch-completed.txt
            |-- num-model-parameters.txt
            |-- saved_model*
            +-- session-info.txt
</pre>

**Run `metrics`, `predict`, `roc`, and `pr` evaluators on model that was previously trained on synthesized data:**
```
seqgra -d DATA_CONFIG_FILE \
       -m MODEL_CONFIG_FILE \
       -e metrics predict roc pr \
       -o OUTPUT_DIR
```

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUT_DIR}</i>
|-- <i>input</i>
|   +-- <i>{GRAMMAR ID}</i>
|       |-- <i>session-info.txt</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>training-grammar-heatmap.txt</i>
|       |-- <i>training-grammar-heatmap.pdf</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>validation-grammar-heatmap.txt</i>
|       |-- <i>validation-grammar-heatmap.pdf</i>
|       |-- <i>test.txt</i>
|       |-- <i>test-annotation.txt</i>
|       |-- <i>test-grammar-heatmap.txt</i>
|       +-- <i>test-grammar-heatmap.pdf</i>
|-- <i>models</i>
|   +-- <i>{GRAMMAR ID}</i>
|       +-- <i>{MODEL ID}</i>
|           |-- <i>last-epoch-completed.txt</i>
|           |-- <i>num-model-parameters.txt</i>
|           |-- <i>saved_model*</i>
|           +-- <i>session-info.txt</i>
+-- evaluation
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            |-- metrics
            |   +-- test-metrics.txt
            |-- predict
            |   +-- test-y-hat.txt
            |-- roc
            |   +-- test-roc-curve.pdf
            +-- pr
                +-- test-pr-curve.pdf
</pre>

**Run SIS evaluator on model that was previously trained on experimental data:**
```
seqgra -f DATA_FOLDER \
       -m MODEL_CONFIG_FILE \
       -e sis \
       -o OUTPUT_DIR
       --eval-n-per-label 30
```

`eval-n-per-label` restricts the number of examples that are evaluated with
SIS to 30 per label. Otherwise sufficient input subsets will be identified
for all examples in the test set, which might take a long time.

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUT_DIR}</i>
|-- <i>input</i>
|   +-- <i>{DATA_FOLDER}</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>test.txt</i>
|       +-- <i>test-annotation.txt</i>
|-- <i>models</i>
|   +-- <i>{GRAMMAR ID}</i>
|       +-- <i>{MODEL ID}</i>
|           |-- <i>last-epoch-completed.txt</i>
|           |-- <i>num-model-parameters.txt</i>
|           |-- <i>saved_model*</i>
|           +-- <i>session-info.txt</i>
+-- evaluation
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            +-- sis
                |-- test-df.txt
                |-- test-grammar-agreement-thresholded.pdf
                |-- test-grammar-agreement-thresholded-df.txt
                +-- test-statistics-thresholded.txt
</pre>

\* model files are library-dependent


**Run various gradient-based evaluation method on model, which was previously trained on experimental data:**
```
seqgra -f DATA_FOLDER \
       -m MODEL_CONFIG_FILE \
       -e gradient gradient-x-input integrated-gradients saliency \
       -o OUTPUT_DIR
       --eval-sets validation test
       --eval-n-per-label 500
```

- the `-e` argument is used to specify a list of evaluators (by their IDs)
- `--eval-sets` selects training, validation or test set for evaluation. Here we run evaluators on both validation and test set examples, default value is test set only.
- `--eval-n-per-label` restricts the number of examples that the evaluators see. Here we evaluate 500 randomly select examples per label. 

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUT_DIR}</i>
|-- <i>input</i>
|   +-- <i>{DATA_FOLDER}</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>test.txt</i>
|       +-- <i>test-annotation.txt</i>
|-- <i>models</i>
|   +-- <i>{GRAMMAR ID}</i>
|       +-- <i>{MODEL ID}</i>
|           |-- <i>last-epoch-completed.txt</i>
|           |-- <i>num-model-parameters.txt</i>
|           |-- <i>saved_model*</i>
|           +-- <i>session-info.txt</i>
+-- evaluation
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            |-- gradient
            |   |-- test-df.txt
            |   |-- test-feature-importance-matrix.npy
            |   |-- test-grammar-agreement-thresholded.pdf
            |   |-- test-grammar-agreement-thresholded-df.txt
            |   |-- test-statistics.txt
            |   |-- test-statistics-thresholded.txt
            |   |-- validation-df.txt
            |   |-- validation-feature-importance-matrix.npy
            |   |-- validation-grammar-agreement-thresholded.pdf
            |   |-- validation-grammar-agreement-thresholded-df.txt
            |   |-- validation-statistics.txt
            |   +-- validation-statistics-thresholded.txt
            |-- gradient-x-input
            |   |-- test-df.txt
            |   |-- test-feature-importance-matrix.npy
            |   |-- test-grammar-agreement-thresholded.pdf
            |   |-- test-grammar-agreement-thresholded-df.txt
            |   |-- test-statistics.txt
            |   |-- test-statistics-thresholded.txt
            |   |-- validation-df.txt
            |   |-- validation-feature-importance-matrix.npy
            |   |-- validation-grammar-agreement-thresholded.pdf
            |   |-- validation-grammar-agreement-thresholded-df.txt
            |   |-- validation-statistics.txt
            |   +-- validation-statistics-thresholded.txt
            |-- integrated-gradients
            |   |-- test-df.txt
            |   |-- test-feature-importance-matrix.npy
            |   |-- test-grammar-agreement-thresholded.pdf
            |   |-- test-grammar-agreement-thresholded-df.txt
            |   |-- test-statistics.txt
            |   |-- test-statistics-thresholded.txt
            |   |-- validation-df.txt
            |   |-- validation-feature-importance-matrix.npy
            |   |-- validation-grammar-agreement-thresholded.pdf
            |   |-- validation-grammar-agreement-thresholded-df.txt
            |   |-- validation-statistics.txt
            |   +-- validation-statistics-thresholded.txt
            +-- saliency
                |-- test-df.txt
                |-- test-feature-importance-matrix.npy
                |-- test-grammar-agreement-thresholded.pdf
                |-- test-grammar-agreement-thresholded-df.txt
                |-- test-statistics.txt
                |-- test-statistics-thresholded.txt
                |-- validation-df.txt
                |-- validation-feature-importance-matrix.npy
                |-- validation-grammar-agreement-thresholded.pdf
                |-- validation-grammar-agreement-thresholded-df.txt
                |-- validation-statistics.txt
                +-- validation-statistics-thresholded.txt
</pre>

\* model files are library-dependent
