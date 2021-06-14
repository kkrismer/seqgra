Usage examples
==============

Most use cases are covered by the seqgra CLI commands ``seqgra``, ``seqgras``,
``seqgrae``, and ``seqgraa``.

``seqgra`` command - core functionality:
    covers the core functionality of simulating data, creating 
    models, training models, saving and loading models, evaluating models using
    conventional test set metrics and feature attribution methods

``seqgras`` command - seqgra summary:
    gathers metrics across grammars, 
    models, evaluators

``seqgrae`` command - seqgra ensemble:
    tests model architecture on grammar 
    across various data set sizes, simulation and model seeds

``seqgraa`` command - seqgra attribution:
    used to obtain feature 
    attribution/evidence for selected examples across multiple grammars, 
    models, evaluators

The following schematic shows various seqgra analyses with inputs, outputs, 
and corresponding commands:

.. image:: _static/seqgra-variants.svg
    :width: 100%
    :alt: seqgra variants

For a detailed description of the four seqgra commands and all arguments,
see :doc:`cmd`.

Commonly used suite of seqgra commands
--------------------------------------

.. code-block:: shell

    seqgra -d DATA_DEFINITION_FILE \
        -m MODEL_DEFINITION_FILE \
        -o OUTPUT_DIR
    seqgra -d DATA_DEFINITION_FILE \
        -m MODEL_DEFINITION_FILE \
        -o OUTPUT_DIR \
        -e metrics roc pr predict \
        --eval-sets training validation test
    seqgra -d DATA_DEFINITION_FILE \
        -m MODEL_DEFINITION_FILE \
        -o OUTPUT_DIR \
        -e sis \
        --eval-n-per-label 20
    seqgra -d DATA_DEFINITION_FILE \
        -m MODEL_DEFINITION_FILE \
        -o OUTPUT_DIR \
        -e gradient saliency gradient-x-input integrated-gradients \
        --eval-n-per-label 50

#. ``seqgra`` call: generate synthetic data as defined in 
   ``DATA_DEFINITION_FILE`` and create model as defined in 
   ``MODEL_DEFINITION_FILE`` and train it on synthetic data
#. ``seqgra`` call: load previously trained model, call conventional 
   evaluators (metrics, roc, pr, and predict) on all examples of training, 
   validation, and test set
#. ``seqgra`` call: load previously trained model, call SIS evaluator on 20 
   test set examples per label (SIS is the most computationally expensive 
   evaluator)
#. ``seqgra`` call: load previously trained model, call gradient-based 
   evaluators (gradient, saliency, gradient-x-input, and 
   integrated-gradients) on 50 test set examples per label

seqgra use cases
----------------

**Input placeholders:**

- ``DATA_DEFINITION_FILE``: path to data definition XML file (see :doc:`dd` 
  for a detailed description of the data definition language and dd-folder_ 
  for examples of data definitions.
- ``MODEL_DEFINITION_FILE``: path to model definition XML file (see :doc:`md` 
  for a detailed description of the model definition language and md-folder_ 
  for examples of model definitions.
- ``OUTPUT``: output folder name

For a detailed description of the arguments, see :doc:`cmd`.

.. note::
    Not all data definition / model definition pairs are valid (i.e.,
    not all architectures can be trained on all data sets): The sequence 
    window (``x``) and the labels/classes (``y``) of the data
    definition have to be compatible with the model definition. E.g., 
    data definitions of task *multi-class classification* can only be paired 
    with model definitions of the same task. Likewise, the number of
    labels/classes and the sequence window width must match between data 
    definition and model definition.

Generate synthetic data only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-s.svg
    :width: 100%
    :alt: seqgra variant - synthetic data only

**Command:**

.. code-block:: shell

    seqgra -d DATA_DEFINITION_FILE \
        -o OUTPUT_DIR

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    +-- input
        +-- {GRAMMAR ID}
            |-- motif-ess-matrix.pdf
            |-- motif-ess-matrix.txt
            |-- motif-ess-se1-violin.pdf
            |-- motif-ess-se2-violin.pdf
            |-- motif-ess-statistics.txt
            |-- motif-info.txt
            |-- motif-kld-matrix.pdf
            |-- motif-kld-matrix.txt
            |-- motif-kld-se1-violin.pdf
            |-- motif-kld-se2-violin.pdf
            |-- motif-kld-statistics.txt
            |-- session-info.txt
            |-- test.txt
            |-- test-annotation.txt
            |-- test-grammar-heatmap.txt
            |-- test-grammar-heatmap.pdf
            |-- training.txt
            |-- training-annotation.txt
            |-- training-grammar-heatmap.txt
            |-- training-grammar-heatmap.pdf
            |-- validation.txt
            |-- validation-annotation.txt
            |-- validation-grammar-heatmap.txt
            +-- validation-grammar-heatmap.pdf

**Pre-existing folders and files:**

- ``DATA_DEFINITION_FILE``

Generate synthetic data and train model on it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-sl.svg
    :width: 100%
    :alt: seqgra variant - generate synthetic data, train model on it

**Command:**

.. code-block:: shell

    seqgra -d DATA_DEFINITION_FILE \
        -m MODEL_DEFINITION_FILE \
        -o OUTPUT_DIR

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    |-- input
    |   +-- {GRAMMAR ID}
    |       |-- motif-ess-matrix.pdf
    |       |-- motif-ess-matrix.txt
    |       |-- motif-ess-se1-violin.pdf
    |       |-- motif-ess-se2-violin.pdf
    |       |-- motif-ess-statistics.txt
    |       |-- motif-info.txt
    |       |-- motif-kld-matrix.pdf
    |       |-- motif-kld-matrix.txt
    |       |-- motif-kld-se1-violin.pdf
    |       |-- motif-kld-se2-violin.pdf
    |       |-- motif-kld-statistics.txt
    |       |-- session-info.txt
    |       |-- test.txt
    |       |-- test-annotation.txt
    |       |-- test-grammar-heatmap.txt
    |       |-- test-grammar-heatmap.pdf
    |       |-- training.txt
    |       |-- training-annotation.txt
    |       |-- training-grammar-heatmap.txt
    |       |-- training-grammar-heatmap.pdf
    |       |-- validation.txt
    |       |-- validation-annotation.txt
    |       |-- validation-grammar-heatmap.txt
    |       +-- validation-grammar-heatmap.pdf
    +-- models
        +-- {GRAMMAR ID}
            +-- {MODEL ID}
                |-- last-epoch-completed.txt
                |-- num-model-parameters.txt
                |-- saved_model†
                +-- session-info.txt

**Pre-existing folders and files:**

- ``DATA_DEFINITION_FILE``
- ``MODEL_DEFINITION_FILE``

Train model on previously synthesized data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-l.svg
    :width: 100%
    :alt: seqgra variant - train model on previously synthesized data

**Command:**

.. code-block:: shell

    seqgra -d DATA_DEFINITION_FILE \
        -m MODEL_DEFINITION_FILE \
        -o OUTPUT_DIR

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    +-- models
        +-- {GRAMMAR ID}
            +-- {MODEL ID}
                |-- last-epoch-completed.txt
                |-- num-model-parameters.txt
                |-- saved_model†
                +-- session-info.txt

**Pre-existing folders and files:**

- ``DATA_DEFINITION_FILE``
- ``MODEL_DEFINITION_FILE``
- ``{OUTPUT_DIR}/input/{GRAMMAR ID}/*``

Train model on experimental or externally synthesized data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-el.svg
    :width: 100%
    :alt: seqgra variant - train model on experimental data

**Command:**

.. code-block:: shell

    seqgra -f DATA_FOLDER \
        -m MODEL_DEFINITION_FILE \
        -o OUTPUT_DIR

where experimental or externally synthesized data is in the 
``{OUTPUT_DIR}/input/{DATA_FOLDER}`` folder.

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    +-- models
        +-- {DATA_FOLDER}
            +-- {MODEL ID}
                |-- last-epoch-completed.txt
                |-- num-model-parameters.txt
                |-- saved_model†
                +-- session-info.txt

**Pre-existing folders and files:**

- ``MODEL_DEFINITION_FILE``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/test.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/test-annotation.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/training.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/training-annotation.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/validation.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/validation-annotation.txt``

Run ``metrics``, ``predict``, ``roc``, and ``pr`` evaluators on model that was previously trained on synthesized data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-e.svg
    :width: 100%
    :alt: seqgra variant - conventional evaluators

**Command:**

.. code-block:: shell

    seqgra -d DATA_DEFINITION_FILE \
        -m MODEL_DEFINITION_FILE \
        -e metrics predict roc pr \
        -o OUTPUT_DIR

.. note::
    - the ``-e`` argument is used to specify a list of evaluators
      by their IDs (see :doc:`slec` for a table of all evaluator IDs)

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    +-- evaluation
        +-- {GRAMMAR ID}
            +-- {MODEL ID}
                |-- metrics
                |   +-- test-metrics.txt
                |-- pr
                |   +-- test-pr-curve.pdf
                |-- predict
                |   +-- test-y-hat.txt
                +-- roc
                    +-- test-roc-curve.pdf

**Pre-existing folders and files:**

- ``DATA_DEFINITION_FILE``
- ``MODEL_DEFINITION_FILE``
- ``{OUTPUT_DIR}/input/{GRAMMAR ID}/*``
- ``{OUTPUT_DIR}/models/{GRAMMAR ID}/{MODEL ID}/*``

Run SIS evaluator on model that was previously trained on experimental data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-sis.svg
    :width: 100%
    :alt: seqgra variant - SIS evaluator

**Command:**

.. code-block:: shell

    seqgra -f DATA_FOLDER \
        -m MODEL_DEFINITION_FILE \
        -e sis \
        -o OUTPUT_DIR \
        --eval-n-per-label 30

.. note::
    - the ``-e`` argument is used to specify a list of evaluators
      by their IDs (see :doc:`slec` for a table of all evaluator IDs)
    - ``--eval-n-per-label 30`` restricts the number of examples that are 
      evaluated with SIS to 30 per label. Otherwise sufficient input subsets 
      will be identified for all examples in the test set, which might take a 
      long time.

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    +-- evaluation
        +-- {DATA_FOLDER}
            +-- {MODEL ID}
                +-- sis
                    |-- test-df.txt
                    |-- test-grammar-agreement-thresholded.pdf
                    |-- test-grammar-agreement-thresholded-df.txt
                    +-- test-statistics-thresholded.txt

**Pre-existing folders and files:**

- ``MODEL_DEFINITION_FILE``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/test.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/test-annotation.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/training.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/training-annotation.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/validation.txt``
- ``{OUTPUT_DIR}/input/{DATA_FOLDER}/validation-annotation.txt``
- ``{OUTPUT_DIR}/models/{DATA_FOLDER}/{MODEL ID}/*``

Generate synthetic data, train model on it, and evaluate model using various gradient-based feature importance evaluators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-slfie.svg
    :width: 100%
    :alt: seqgra variant - FIE evaluators

**Command:**

.. code-block:: shell

    seqgra -d DATA_DEFINITION_FILE \
        -m MODEL_DEFINITION_FILE \
        -e gradient gradient-x-input integrated-gradients saliency \
        -o OUTPUT_DIR \
        --eval-sets validation test \
        --eval-n-per-label 500

.. note::
    - the ``-e`` argument is used to specify a list of evaluators
      by their IDs (see :doc:`slec` for a table of all evaluator IDs)
    - ``--eval-sets`` selects training, validation or test set for evaluation. 
      Here we run evaluators on both validation and test set examples, 
      default value is test set only.
    - ``--eval-n-per-label`` restricts the number of examples that 
      the evaluators see. Here we evaluate 500 randomly select examples 
      per label.

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    |-- input
    |   +-- {GRAMMAR ID}
    |       |-- motif-ess-matrix.pdf
    |       |-- motif-ess-matrix.txt
    |       |-- motif-ess-se1-violin.pdf
    |       |-- motif-ess-se2-violin.pdf
    |       |-- motif-ess-statistics.txt
    |       |-- motif-info.txt
    |       |-- motif-kld-matrix.pdf
    |       |-- motif-kld-matrix.txt
    |       |-- motif-kld-se1-violin.pdf
    |       |-- motif-kld-se2-violin.pdf
    |       |-- motif-kld-statistics.txt
    |       |-- session-info.txt
    |       |-- test.txt
    |       |-- test-annotation.txt
    |       |-- test-grammar-heatmap.txt
    |       |-- test-grammar-heatmap.pdf
    |       |-- training.txt
    |       |-- training-annotation.txt
    |       |-- training-grammar-heatmap.txt
    |       |-- training-grammar-heatmap.pdf
    |       |-- validation.txt
    |       |-- validation-annotation.txt
    |       |-- validation-grammar-heatmap.txt
    |       +-- validation-grammar-heatmap.pdf
    |-- models
    |   +-- {GRAMMAR ID}
    |       +-- {MODEL ID}
    |           |-- last-epoch-completed.txt
    |           |-- num-model-parameters.txt
    |           |-- saved_model†
    |           +-- session-info.txt
    +-- evaluation
        +-- {GRAMMAR ID}
            +-- {MODEL ID}
                |-- gradient
                |   |-- test-df.txt
                |   |-- test-feature-importance-matrix.npy
                |   |-- test-grammar-agreement.pdf
                |   |-- test-grammar-agreement-df.txt
                |   |-- test-grammar-agreement-thresholded.pdf
                |   |-- test-grammar-agreement-thresholded-df.txt
                |   |-- test-statistics.txt
                |   |-- test-statistics-thresholded.txt
                |   |-- validation-df.txt
                |   |-- validation-feature-importance-matrix.npy
                |   |-- validation-grammar-agreement.pdf
                |   |-- validation-grammar-agreement-df.txt
                |   |-- validation-grammar-agreement-thresholded.pdf
                |   |-- validation-grammar-agreement-thresholded-df.txt
                |   |-- validation-statistics.txt
                |   +-- validation-statistics-thresholded.txt
                |-- gradient-x-input
                |   |-- test-df.txt
                |   |-- test-feature-importance-matrix.npy
                |   |-- test-grammar-agreement.pdf
                |   |-- test-grammar-agreement-df.txt
                |   |-- test-grammar-agreement-thresholded.pdf
                |   |-- test-grammar-agreement-thresholded-df.txt
                |   |-- test-statistics.txt
                |   |-- test-statistics-thresholded.txt
                |   |-- validation-df.txt
                |   |-- validation-feature-importance-matrix.npy
                |   |-- validation-grammar-agreement.pdf
                |   |-- validation-grammar-agreement-df.txt
                |   |-- validation-grammar-agreement-thresholded.pdf
                |   |-- validation-grammar-agreement-thresholded-df.txt
                |   |-- validation-statistics.txt
                |   +-- validation-statistics-thresholded.txt
                |-- integrated-gradients
                |   |-- test-df.txt
                |   |-- test-feature-importance-matrix.npy
                |   |-- test-grammar-agreement.pdf
                |   |-- test-grammar-agreement-df.txt
                |   |-- test-grammar-agreement-thresholded.pdf
                |   |-- test-grammar-agreement-thresholded-df.txt
                |   |-- test-statistics.txt
                |   |-- test-statistics-thresholded.txt
                |   |-- validation-df.txt
                |   |-- validation-feature-importance-matrix.npy
                |   |-- validation-grammar-agreement.pdf
                |   |-- validation-grammar-agreement-df.txt
                |   |-- validation-grammar-agreement-thresholded.pdf
                |   |-- validation-grammar-agreement-thresholded-df.txt
                |   |-- validation-statistics.txt
                |   +-- validation-statistics-thresholded.txt
                +-- saliency
                    |-- test-df.txt
                    |-- test-feature-importance-matrix.npy
                    |-- test-grammar-agreement.pdf
                    |-- test-grammar-agreement-df.txt
                    |-- test-grammar-agreement-thresholded.pdf
                    |-- test-grammar-agreement-thresholded-df.txt
                    |-- test-statistics.txt
                    |-- test-statistics-thresholded.txt
                    |-- validation-df.txt
                    |-- validation-feature-importance-matrix.npy
                    |-- validation-grammar-agreement.pdf
                    |-- validation-grammar-agreement-df.txt
                    |-- validation-grammar-agreement-thresholded.pdf
                    |-- validation-grammar-agreement-thresholded-df.txt
                    |-- validation-statistics.txt
                    +-- validation-statistics-thresholded.txt

**Pre-existing folders and files:**

- ``DATA_DEFINITION_FILE``
- ``MODEL_DEFINITION_FILE``

Generate collection of data definitions and model definitions derived from root definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-seqgrae.svg
    :width: 100%
    :alt: seqgra variant - seqgrae

This command is used to generate data definitions with various simulation 
seeds and data set sizes and model definition with various model seeds.

**Command:**

.. code-block:: shell

    seqgrae -a ANALYSIS_ID \
        -d DATA_DEFINITION_FILE
        -m MODEL_DEFINITION_FILE_1 MODEL_DEFINITION_FILE_2
        -o OUTPUT_DIR

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    |-- defs
    |   |-- data
    |   |   |-- DATA_DEFINITION_FILE-10k-s1.xml
    |   |   |-- DATA_DEFINITION_FILE-10k-s2.xml
    |   |   |-- DATA_DEFINITION_FILE-10k-s3.xml
    |   |   |-- DATA_DEFINITION_FILE-20k-s1.xml
    |   |   |-- DATA_DEFINITION_FILE-20k-s2.xml
    |   |   |-- DATA_DEFINITION_FILE-20k-s3.xml
    |   |   |-- DATA_DEFINITION_FILE-40k-s1.xml
    |   |   |-- DATA_DEFINITION_FILE-40k-s2.xml
    |   |   |-- DATA_DEFINITION_FILE-40k-s3.xml
    |   |   |-- DATA_DEFINITION_FILE-80k-s1.xml
    |   |   |-- DATA_DEFINITION_FILE-80k-s2.xml
    |   |   |-- DATA_DEFINITION_FILE-80k-s3.xml
    |   |   |-- DATA_DEFINITION_FILE-160k-s1.xml
    |   |   |-- DATA_DEFINITION_FILE-160k-s2.xml
    |   |   |-- DATA_DEFINITION_FILE-160k-s3.xml
    |   |   |-- DATA_DEFINITION_FILE-320k-s1.xml
    |   |   |-- DATA_DEFINITION_FILE-320k-s2.xml
    |   |   |-- DATA_DEFINITION_FILE-320k-s3.xml
    |   |   |-- DATA_DEFINITION_FILE-640k-s1.xml
    |   |   |-- DATA_DEFINITION_FILE-640k-s2.xml
    |   |   |-- DATA_DEFINITION_FILE-640k-s3.xml
    |   |   |-- DATA_DEFINITION_FILE-1280k-s1.xml
    |   |   |-- DATA_DEFINITION_FILE-1280k-s2.xml
    |   |   |-- DATA_DEFINITION_FILE-1280k-s3.xml
    |   +-- model
    |       |-- MODEL_DEFINITION_FILE_1-s1.xml
    |       |-- MODEL_DEFINITION_FILE_1-s2.xml
    |       |-- MODEL_DEFINITION_FILE_1-s3.xml
    |       |-- MODEL_DEFINITION_FILE_2-s1.xml
    |       |-- MODEL_DEFINITION_FILE_2-s2.xml
    |       +-- MODEL_DEFINITION_FILE_2-s3.xml
    +-- analyses
        +-- {ANALYSIS ID}.sh

**Pre-existing folders and files:**

- ``DATA_DEFINITION_FILE``
- ``MODEL_DEFINITION_FILE_1``
- ``MODEL_DEFINITION_FILE_2``

Subsample experimental data and generate collection of model definitions derived from root definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-seqgrae-exp.svg
    :width: 100%
    :alt: seqgra variant - seqgrae on experimental data

This command is used to subsample experimental data and generate model 
definition with various model seeds.

**Command:**

.. code-block:: shell

    seqgrae -a ANALYSIS_ID \
        -f DATA_FOLDER \
        -m MODEL_DEFINITION_FILE \
        -o OUTPUT_DIR

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    |-- defs
    |   +-- model
    |       |-- MODEL_DEFINITION_FILE-s1.xml
    |       |-- MODEL_DEFINITION_FILE-s2.xml
    |       |-- MODEL_DEFINITION_FILE-s3.xml
    |-- inputs
    |   |-- DATA_FOLDER-0.05-s1.xml
    |   |-- DATA_FOLDER-0.05-s2.xml
    |   |-- DATA_FOLDER-0.05-s3.xml
    |   |-- DATA_FOLDER-0.1-s1.xml
    |   |-- DATA_FOLDER-0.1-s2.xml
    |   |-- DATA_FOLDER-0.1-s3.xml
    |   |-- DATA_FOLDER-0.2-s1.xml
    |   |-- DATA_FOLDER-0.2-s2.xml
    |   |-- DATA_FOLDER-0.2-s3.xml
    |   |-- DATA_FOLDER-0.4-s1.xml
    |   |-- DATA_FOLDER-0.4-s2.xml
    |   |-- DATA_FOLDER-0.4-s3.xml
    |   |-- DATA_FOLDER-0.8-s1.xml
    |   |-- DATA_FOLDER-0.8-s2.xml
    |   |-- DATA_FOLDER-0.8-s3.xml
    |   |-- DATA_FOLDER-1.0-s1.xml
    |   |-- DATA_FOLDER-1.0-s2.xml
    |   |-- DATA_FOLDER-1.0-s3.xml
    +-- analyses
        +-- {ANALYSIS ID}.sh

**Pre-existing folders and files:**

- ``DATA_FOLDER``
- ``MODEL_DEFINITION_FILE``

Summarize results across multiple grammars using comparators ``roc`` and ``pr``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-seqgras.svg
    :width: 100%
    :alt: seqgra variant - seqgras ROC, PR

**Command:**

.. code-block:: shell

    seqgras -a sim-basic-mc2-tf-mc2-dna1000-conv10-fc5 \
        -c roc pr \
        -o data \
        -g mc2-dna1000-homer-10k-s1 mc2-dna1000-homer-20k-s1 mc2-dna1000-homer-30k-s1 \
            mc2-dna1000-homer-40k-s1 mc2-dna1000-homer-50k-s1 mc2-dna1000-homer-60k-s1 \
            mc2-dna1000-homer-70k-s1 mc2-dna1000-homer-80k-s1 mc2-dna1000-homer-90k-s1 \
            mc2-dna1000-homer-100k-s1 mc2-dna1000-homer-110k-s1 mc2-dna1000-homer-120k-s1 \
            mc2-dna1000-homer-130k-s1 mc2-dna1000-homer-140k-s1 mc2-dna1000-homer-150k-s1 \
            mc2-dna1000-homer-200k-s1 mc2-dna1000-homer-500k-s1 mc2-dna1000-homer-1000k-s1 \
            mc2-dna1000-homer-2000k-s1 \
        -m tf-mc2-dna1000-conv10-fc5 \
        -l '10,000 examples' '20,000 examples' '30,000 examples' '40,000 examples' \
            '50,000 examples' '60,000 examples' '70,000 examples' '80,000 examples' \
            '90,000 examples' '100,000 examples' '110,000 examples' '120,000 examples' \
            '130,000 examples' '140,000 examples' '150,000 examples' '200,000 examples' \
            '500,000 examples' '1,000,000 examples' '2,000,000 examples'

.. note::
    - the ``-c`` argument is used to specify a list of comparators
      by their IDs (see :doc:`slec` for a table of all comparator IDs)
    - the ``-g`` argument is used to specify all grammar IDs / data folders
    - the ``-m`` argument is used to specify all model IDs
    - the ``-l`` argument is used to label the curves for ROC/PR comparators

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    +-- model-comparisons
        +-- sim-basic-mc2-tf-mc2-dna1000-conv10-fc5
            |-- test-pr-curve.pdf
            +-- test-roc-curve.pdf

**Pre-existing folders and files:**

- ``{OUTPUT_DIR}/input/{mc2-dna1000-homer-10k}/*``
- ``{OUTPUT_DIR}/models/{mc2-dna1000-homer-10k}/{tf-mc2-dna1000-conv10-fc5}/*``
- ``{OUTPUT_DIR}/evaluation/{mc2-dna1000-homer-10k}/{tf-mc2-dna1000-conv10-fc5}/*``
- ``{OUTPUT_DIR}/input/{mc2-dna1000-homer-20k}/*``
- ``{OUTPUT_DIR}/models/{mc2-dna1000-homer-20k}/{tf-mc2-dna1000-conv10-fc5}/*``
- ``{OUTPUT_DIR}/evaluation/{mc2-dna1000-homer-20k}/{tf-mc2-dna1000-conv10-fc5}/*``
- …

Summarize results across multiple grammars using comparators ``table``, ``curve-table``, and ``fi-eval-table``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/seqgra-variants-seqgras.svg
    :width: 100%
    :alt: seqgra variant - seqgras Table/Curve-Table/FIE-Table

**Command:**

.. code-block:: shell

    seqgras -a sim-basic-mc10-interaction-spacing-torch-mc10-dna1000-conv10w-gmp-fc10
        -c table curve-table fi-eval-table
        -o data
        -g mc10-dna1000-homer-interaction-spacing-10k \
            mc10-dna1000-homer-interaction-spacing-10k-1 \
            mc10-dna1000-homer-interaction-spacing-10k-2 \
            mc10-dna1000-homer-interaction-spacing-10k-3 \
            mc10-dna1000-homer-interaction-spacing-10k-4 \
            mc10-dna1000-homer-interaction-spacing-20k \
            mc10-dna1000-homer-interaction-spacing-20k-1 \
            mc10-dna1000-homer-interaction-spacing-20k-2 \
            mc10-dna1000-homer-interaction-spacing-20k-3 \
            mc10-dna1000-homer-interaction-spacing-20k-4 \
            mc10-dna1000-homer-interaction-spacing-30k \
            mc10-dna1000-homer-interaction-spacing-30k-1 \
            mc10-dna1000-homer-interaction-spacing-30k-2 \
            mc10-dna1000-homer-interaction-spacing-30k-3 \
            mc10-dna1000-homer-interaction-spacing-30k-4 \
            mc10-dna1000-homer-interaction-spacing-40k \
            mc10-dna1000-homer-interaction-spacing-40k-1 \
            mc10-dna1000-homer-interaction-spacing-40k-2 \
            mc10-dna1000-homer-interaction-spacing-40k-3 \
            mc10-dna1000-homer-interaction-spacing-40k-4 \
            mc10-dna1000-homer-interaction-spacing-50k \
            mc10-dna1000-homer-interaction-spacing-50k-1 \
            mc10-dna1000-homer-interaction-spacing-50k-2 \
            mc10-dna1000-homer-interaction-spacing-50k-3 \
            mc10-dna1000-homer-interaction-spacing-50k-4 \
            mc10-dna1000-homer-interaction-spacing-60k \
            mc10-dna1000-homer-interaction-spacing-60k-1 \
            mc10-dna1000-homer-interaction-spacing-60k-2 \
            mc10-dna1000-homer-interaction-spacing-60k-3 \
            mc10-dna1000-homer-interaction-spacing-60k-4 \
            mc10-dna1000-homer-interaction-spacing-70k \
            mc10-dna1000-homer-interaction-spacing-70k-1 \
            mc10-dna1000-homer-interaction-spacing-70k-2 \
            mc10-dna1000-homer-interaction-spacing-70k-3 \
            mc10-dna1000-homer-interaction-spacing-70k-4 \
            mc10-dna1000-homer-interaction-spacing-80k \
            mc10-dna1000-homer-interaction-spacing-80k-1 \
            mc10-dna1000-homer-interaction-spacing-80k-2 \
            mc10-dna1000-homer-interaction-spacing-80k-3 \
            mc10-dna1000-homer-interaction-spacing-80k-4 \
            mc10-dna1000-homer-interaction-spacing-90k \
            mc10-dna1000-homer-interaction-spacing-90k-1 \
            mc10-dna1000-homer-interaction-spacing-90k-2 \
            mc10-dna1000-homer-interaction-spacing-90k-3 \
            mc10-dna1000-homer-interaction-spacing-90k-4 \
            mc10-dna1000-homer-interaction-spacing-100k \
            mc10-dna1000-homer-interaction-spacing-100k-1 \
            mc10-dna1000-homer-interaction-spacing-100k-2 \
            mc10-dna1000-homer-interaction-spacing-100k-3 \
            mc10-dna1000-homer-interaction-spacing-100k-4 \
        -m torch-mc10-dna1000-conv10w-gmp-fc10

.. note::
    - the ``-c`` argument is used to specify a list of comparators
      by their IDs (see :doc:`slec` for a table of all comparator IDs)
    - the ``-g`` argument is used to specify all grammar IDs / data folders
    - the ``-m`` argument is used to specify all model IDs

**Generated folders and files:**

.. code-block:: text

    {OUTPUT_DIR}
    +-- model-comparisons
        +-- sim-basic-mc10-interaction-spacing-torch-mc10-dna1000-conv10w-gmp-fc10
            |-- curve-table.txt
            |-- fie-table.txt
            +-- table.txt

**Pre-existing folders and files:**

- ``{OUTPUT_DIR}/input/{mc10-dna1000-homer-interaction-spacing-10k}/*``
- ``{OUTPUT_DIR}/models/{mc10-dna1000-homer-interaction-spacing-10k}/{torch-mc10-dna1000-conv10w-gmp-fc10}/*``
- ``{OUTPUT_DIR}/evaluation/{mc10-dna1000-homer-interaction-spacing-10k}/{torch-mc10-dna1000-conv10w-gmp-fc10}/*``
- ``{OUTPUT_DIR}/input/{mc10-dna1000-homer-interaction-spacing-10k-1}/*``
- ``{OUTPUT_DIR}/models/{mc10-dna1000-homer-interaction-spacing-10k-1}/{torch-mc10-dna1000-conv10w-gmp-fc10}/*``
- ``{OUTPUT_DIR}/evaluation/{mc10-dna1000-homer-interaction-spacing-10k-1}/{torch-mc10-dna1000-conv10w-gmp-fc10}/*``
- …

† model files are library-dependent

.. _dd-folder: https://github.com/kkrismer/seqgra/tree/master/docsrc/defs/dd
.. _md-folder: https://github.com/kkrismer/seqgra/tree/master/docsrc/defs/md