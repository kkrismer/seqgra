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
pip install .
```

## Usage

```
seqgra -h
usage: seqgra [-h] (-d DATACONFIGFILE | -f DATAFOLDER) -m MODELCONFIGFILE
              [-e {sis}] -o OUTPUTDIR

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
  -e {sis}, --evaluator {sis}
                        evaluator ID of interpretability method
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        output directory, subdirectories are created for
                        generated data, trained model, and model evaluation
```

## Citation

## Funding
