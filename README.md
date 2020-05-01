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

For now, download this seqgra repository and call the following pip command 
inside the seqgra root directory:

```
pip install -e .
```

## Usage

```
seqgra -h
usage: seqgra [-h] (-d DATACONFIGFILE | -f DATAFOLDER) [-m MODELCONFIGFILE] 
              [-e EVALUATORS [EVALUATORS ...]] -o OUTPUTDIR 
              [--eval-sets EVAL_SETS [EVAL_SETS ...]] [--eval-n EVAL_N] 
              [--eval-n-per-label EVAL_N_PER_LABEL] 
              [--eval-fi-predict-threshold EVAL_FI_PREDICT_THRESHOLD]
              [--eval-sis-predict-threshold EVAL_SIS_PREDICT_THRESHOLD]

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
```

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
