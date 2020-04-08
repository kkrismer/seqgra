# seqgra: synthetic biological sequence data using sequence grammars

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
              [-e EVALUATOR [EVALUATOR ...]] -o OUTPUTDIR

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
  -e EVALUATOR [EVALUATOR ...], --evaluator EVALUATOR [EVALUATOR ...]
                        evaluator ID or IDs of interpretability method - valid 
                        evaluator IDs include metrics, predict, roc, pr, sis
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        output directory, subdirectories are created for 
                        generated data, trained model, and model evaluation
```

## Types of seqgra analyses

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
where previously synthesized data is in `{OUTPUTDIR}/input/{GRAMMAR ID}` folder and `{GRAMMAR ID}` is defined in `{DATACONFIGFILE}`.

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
where experimental or externally synthesized data is in `{OUTPUTDIR}/input/{DATAFOLDER}` folder.

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

**Run `metrics`, `predict`, `roc`, and `pr` evaluation methods on model, which was previously trained on synthesized data:**
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
            |   +-- metrics.txt
            |-- predict
            |   |-- y-hat-training.txt
            |   |-- y-hat-validation.txt
            |   +-- y-hat-test.txt
            |-- roc
            |   |-- roc-curve-training.pdf
            |   |-- roc-curve-validation.pdf
            |   +-- roc-curve-test.pdf
            +-- pr
                |-- pr-curve-training.pdf
                |-- pr-curve-validation.pdf
                +-- pr-curve-test.pdf
</pre>

**Run SIS evaluation method on model, which was previously trained on experimental data:**
```
seqgra -f DATAFOLDER \
       -m MODELCONFIGFILE \
       -e sis \
       -o OUTPUTDIR
```

Generated files and folders (pre-existing folders and files in gray):
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
                |-- sis-heatmap.pdf
                |-- test-examples-class1.txt
                |-- test-examples-class2.txt
                +-- test-examples-classn.txt
</pre>

\* model files are library-dependent

## Citation

## Funding
