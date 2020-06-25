# seqgra: Synthetic rule-based biological sequence data generation for architecture evaluation and search

*Travis-CI build badge placeholder*

*PyPI version badge placeholder*

*DOI badge placeholder*

## Installation

Ultimately, seqgra will be available on the Python Package Index (PyPI) and 
can be installed with pip:

```
pip install seqgra
```

For now, install seqgra from this repository directly:

```
git clone https://github.com/kkrismer/seqgra
cd seqgra
pip install .
```

### System requirements

- Python 3.7 (or higher)
- *R 3.5 (or higher)*
    - *R package `ggplot2` 3.3.0 (or higher)*
    - *R package `scales` 1.1.0 (or higher)*

R is used to create the grammar-model-agreement plots and if not available,
these plots will be skipped.

### Python package dependencies

- Cython>=0.29
- lxml>=4.4.1
- matplotlib>=3.1
- numpy>=1.14
- pandas>=0.25
- PyYAML>=5.3
- scikit-image>=0.16
- scikit-learn>=0.21
- scipy>=1.3
- setuptools>=41.6
- ushuffle>=1.1.2
- *tensorflow>=2.0.0 (NOT automatically installed by pip)*
- *torch>=1.4.0 (NOT automatically installed by pip)*
- *pytorch-ignite>=0.3.0 (NOT automatically installed by pip)*

The `tensorflow` package is only required if TensorFlow models are used and
will not be automatically installed by `pip install seqgra`. Same is true for
packages `torch` and `pytorch-ignite`, which are only required if PyTorch 
models are used.

## Usage

```
seqgra -h
usage: seqgra [-h] 
              (-d DATACONFIGFILE | -f DATAFOLDER)
              [-m MODELCONFIGFILE]
              [-e EVALUATORS [EVALUATORS ...]]
              -o OUTPUTDIR
              [-p]
              [-r]
              [-g GPU]
              [--nochecks]
              [--eval-sets EVAL_SETS [EVAL_SETS ...]]
              [--eval-n EVAL_N]
              [--eval-n-per-label EVAL_N_PER_LABEL]
              [--eval-suppress-plots]
              [--eval-fi-predict-threshold EVAL_FI_PREDICT_THRESHOLD]
              [--eval-sis-predict-threshold EVAL_SIS_PREDICT_THRESHOLD]
              [--eval-grad-importance-threshold EVAL_GRAD_IMPORTANCE_THRESHOLD]

Generate synthetic data based on grammar, train model on synthetic data, 
evaluate model

optional arguments:
  -h, --help            show this help message and exit
  -d DATACONFIGFILE, --dataconfigfile DATACONFIGFILE
                        path to the segra XML data configuration file. Use 
                        this option to generate synthetic data based on a 
                        seqgra grammar (specify either -d or -f, not both)
  -f DATAFOLDER, --datafolder DATAFOLDER
                        experimental data folder name inside outputdir/input. 
                        Use this option to train the model on experimental or 
                        externally synthesized data (specify either -f or -d, 
                        not both)
  -m MODELCONFIGFILE, --modelconfigfile MODELCONFIGFILE
                        path to the seqgra XML model configuration file
  -e EVALUATORS [EVALUATORS ...], --evaluators EVALUATORS [EVALUATORS ...]
                        evaluator ID or IDs: IDs of conventional evaluators 
                        include metrics, pr, predict, roc; IDs of feature 
                        importance evaluators include 
                        contrastive-excitation-backprop, deep-lift, 
                        excitation-backprop, grad-cam, gradient, 
                        gradient-x-input, integrated-gradients,      
                        nonlinear-integrated-gradients, saliency, sis
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        output directory, subdirectories are created for 
                        generated data, trained model, and model evaluation
  -p, --print           if this flag is set, data definition, model 
                        definition, and model summary are printed
  -r, --remove          if this flag is set, previously stored data for this 
                        grammar - model combination will be removed prior to 
                        the analysis run. This includes the folders 
                        input/[grammar ID], models/[grammar ID]/[model ID], 
                        and evaluation/[grammar ID]/[model ID].
  -g GPU, --gpu GPU     ID of GPU used by TensorFlow and PyTorch (defaults to 
                        GPU ID 0); CPU is used if no GPU is available or
                        GPU ID is set to -1
  --nochecks            if this flag is set, examples and example annotations 
                        will not be validated before training, e.g., that DNA 
                        sequences only contain A, C, G, T, N
  --eval-sets EVAL_SETS [EVAL_SETS ...]
                        either one or more of the following: training, 
                        validation, test; selects data set for evaluation; 
                        this evaluator argument will be passed to all 
                        evaluators
  --eval-n EVAL_N       maximum number of examples to be evaluated per set 
                        (defaults to the total number of examples); this 
                        evaluator argument will be passed to all evaluators
  --eval-n-per-label EVAL_N_PER_LABEL
                        maximum number of examples to be evaluated for each 
                        label and set (defaults to the total number of 
                        examples unless eval-n is set, overrules eval-n); 
                        this evaluator argument will be passed to all 
                        evaluators
  --eval-suppress-plots
                        if this flag is set, plots are suppressed globally; 
                        this evaluator argument will be passed to all 
                        evaluators
  --eval-fi-predict-threshold EVAL_FI_PREDICT_THRESHOLD
                        prediction threshold used to select examples for 
                        evaluation, only examples with predict(x) > threshold 
                        will be passed on to evaluators (defaults to 0.5); 
                        this evaluator argument will be passed to feature 
                        importance evaluators only
  --eval-sis-predict-threshold EVAL_SIS_PREDICT_THRESHOLD
                        prediction threshold for Sufficient Input Subsets; 
                        this evaluator argument is only visible to the SIS 
                        evaluator
  --eval-grad-importance-threshold EVAL_GRAD_IMPORTANCE_THRESHOLD
                        feature importance threshold for gradient-based 
                        feature importance evaluators; this parameter only 
                        affects thresholded grammar agreement plots, not the 
                        feature importance measures themselves; this evaluator 
                        argument is only visible to gradient-based feature     
                        importance evaluators (defaults to 0.01)
```

## Commonly used suite of seqgra commands

```
seqgra -d DATACONFIGFILE \
       -m MODELCONFIGFILE \
       -o OUTPUTDIR
seqgra -d DATACONFIGFILE \
       -m MODELCONFIGFILE
       -o OUTPUTDIR
       -e metrics roc pr predict
       --eval-sets training validation test
seqgra -d DATACONFIGFILE
       -m MODELCONFIGFILE
       -o OUTPUTDIR
       -e sis
       --eval-n-per-label 20
seqgra -d DATACONFIGFILE
       -m MODELCONFIGFILE
       -o OUTPUTDIR
       -e gradient saliency gradient-x-input integrated-gradients
       --eval-n-per-label 50
```

1. generate synthetic data and train model on it
2. load previously trained model, call conventional evaluators (`metrics`, `roc`, `pr`, and `predict`) on all examples of training, validation, and test set
3. load previously trained model, call SIS evaluator on 20 test set examples per label (SIS is the most computationally expensive evaluator)
4. load previously trained model, call gradient-based evaluators (`gradient`, `saliency`, `gradient-x-input`, and `integrated-gradients`) on 50 test set examples per label

## Examples of seqgra analyses

**Generate synthetic data only:**
```
seqgra -d DATACONFIGFILE \
       -o OUTPUTDIR
```

Generated files and folders:
<pre>
{OUTPUTDIR}
+-- input
    +-- {GRAMMAR ID}
        |-- session-info.txt
        |-- training.txt
        |-- training-annotation.txt
        |-- validation.txt
        |-- validation-annotation.txt
        |-- test.txt
        +-- test-annotation.txt
</pre>

**Generate synthetic data and train model on it:**
```
seqgra -d DATACONFIGFILE \
       -m MODELCONFIGFILE \
       -o OUTPUTDIR
```

Generated files and folders:
<pre>
{OUTPUTDIR}
|-- input
|   +-- {GRAMMAR ID}
|       |-- session-info.txt
|       |-- training.txt
|       |-- training-annotation.txt
|       |-- validation.txt
|       |-- validation-annotation.txt
|       |-- test.txt
|       +-- test-annotation.txt
+-- models
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            |-- session-info.txt
            +-- saved_model*
</pre>

**Train model on previously synthesized data:**
```
seqgra -d DATACONFIGFILE \
       -m MODELCONFIGFILE \
       -o OUTPUTDIR
```
where previously synthesized data is in the `{OUTPUTDIR}/input/{GRAMMAR ID}` 
folder and `{GRAMMAR ID}` is defined in `{DATACONFIGFILE}`.

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUTDIR}</i>
|-- <i>input</i>
|   +-- <i>{GRAMMAR ID}</i>
|       |-- <i>session-info.txt</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>test.txt</i>
|       +-- <i>test-annotation.txt</i>
+-- models
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            |-- session-info.txt
            +-- saved_model*
</pre>

**Train model on experimental or externally synthesized data:**
```
seqgra -f DATAFOLDER \
       -m MODELCONFIGFILE \
       -o OUTPUTDIR
```
where experimental or externally synthesized data is in the 
`{OUTPUTDIR}/input/{DATAFOLDER}` folder.

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUTDIR}</i>
|-- <i>input</i>
|   +-- <i>{DATAFOLDER}</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>test.txt</i>
|       +-- <i>test-annotation.txt</i>
+-- models
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            |-- session-info.txt
            +-- saved_model*
</pre>

**Run `metrics`, `predict`, `roc`, and `pr` evaluators on model that was previously trained on synthesized data:**
```
seqgra -d DATACONFIGFILE \
       -m MODELCONFIGFILE \
       -e metrics predict roc pr \
       -o OUTPUTDIR
```

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUTDIR}</i>
|-- <i>input</i>
|   +-- <i>{GRAMMAR ID}</i>
|       |-- <i>session-info.txt</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>test.txt</i>
|       +-- <i>test-annotation.txt</i>
|-- <i>models</i>
|   +-- <i>{GRAMMAR ID}</i>
|       +-- <i>{MODEL ID}</i>
|           |-- <i>session-info.txt</i>
|           +-- <i>saved_model*</i>
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
seqgra -f DATAFOLDER \
       -m MODELCONFIGFILE \
       -e sis \
       -o OUTPUTDIR
       --eval-n-per-label 30
```

`eval-n-per-label` restricts the number of examples that are evaluated with
SIS to 30 per label. Otherwise sufficient input subsets will be identified
for all examples in the test set, which might take a long time.

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUTDIR}</i>
|-- <i>input</i>
|   +-- <i>{DATAFOLDER}</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>test.txt</i>
|       +-- <i>test-annotation.txt</i>
|-- <i>models</i>
|   +-- <i>{GRAMMAR ID}</i>
|       +-- <i>{MODEL ID}</i>
|           |-- <i>session-info.txt</i>
|           +-- <i>saved_model*</i>
+-- evaluation
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            +-- sis
                |-- test-df.txt
                |-- test-grammar-agreement-thresholded.pdf
                +-- test-grammar-agreement-thresholded-df.txt
</pre>

\* model files are library-dependent


**Run various gradient-based evaluation method on model, which was previously trained on experimental data:**
```
seqgra -f DATAFOLDER \
       -m MODELCONFIGFILE \
       -e gradient gradient-x-input integrated-gradients saliency \
       -o OUTPUTDIR
       --eval-sets validation test
       --eval-n-per-label 500
```

- the `-e` argument is used to specify a list of evaluators (by their IDs)
- `--eval-sets` selects training, validation or test set for evaluation. Here we run evaluators on both validation and test set examples, default value is test set only.
- `--eval-n-per-label` restricts the number of examples that the evaluators see. Here we evaluate 500 randomly select examples per label. 

Generated files and folders (pre-existing folders and files in italics):
<pre>
<i>{OUTPUTDIR}</i>
|-- <i>input</i>
|   +-- <i>{DATAFOLDER}</i>
|       |-- <i>training.txt</i>
|       |-- <i>training-annotation.txt</i>
|       |-- <i>validation.txt</i>
|       |-- <i>validation-annotation.txt</i>
|       |-- <i>test.txt</i>
|       +-- <i>test-annotation.txt</i>
|-- <i>models</i>
|   +-- <i>{GRAMMAR ID}</i>
|       +-- <i>{MODEL ID}</i>
|           |-- <i>session-info.txt</i>
|           +-- <i>saved_model*</i>
+-- evaluation
    +-- {GRAMMAR ID}
        +-- {MODEL ID}
            |-- gradient
            |   |-- test-df.txt
            |   |-- test-feature-importance-matrix.npy
            |   |-- test-grammar-agreement-thresholded.pdf
            |   |-- test-grammar-agreement-thresholded-df.txt
            |   |-- validation-df.txt
            |   |-- validation-feature-importance-matrix.npy
            |   |-- validation-grammar-agreement-thresholded.pdf
            |   +-- validation-grammar-agreement-thresholded-df.txt
            |-- gradient-x-input
            |   |-- test-df.txt
            |   |-- test-feature-importance-matrix.npy
            |   |-- test-grammar-agreement-thresholded.pdf
            |   |-- test-grammar-agreement-thresholded-df.txt
            |   |-- validation-df.txt
            |   |-- validation-feature-importance-matrix.npy
            |   |-- validation-grammar-agreement-thresholded.pdf
            |   +-- validation-grammar-agreement-thresholded-df.txt
            |-- integrated-gradients
            |   |-- test-df.txt
            |   |-- test-feature-importance-matrix.npy
            |   |-- test-grammar-agreement-thresholded.pdf
            |   |-- test-grammar-agreement-thresholded-df.txt
            |   |-- validation-df.txt
            |   |-- validation-feature-importance-matrix.npy
            |   |-- validation-grammar-agreement-thresholded.pdf
            |   +-- validation-grammar-agreement-thresholded-df.txt
            +-- saliency
                |-- test-df.txt
                |-- test-feature-importance-matrix.npy
                |-- test-grammar-grammar-agreement-thresholded.pdf
                |-- test-grammar-agreement-thresholded-df.txt
                |-- validation-df.txt
                |-- validation-feature-importance-matrix.npy
                |-- validation-grammar-agreement-thresholded.pdf
                +-- validation-grammar-agreement-thresholded-df.txt
</pre>

\* model files are library-dependent

## Citation

## Funding
