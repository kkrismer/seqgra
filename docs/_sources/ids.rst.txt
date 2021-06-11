.. include:: .id_parts.rst

ID conventions
================================

.. note::
    While grammar and model IDs can contain any unique combination of 
    alphanumeric characters (and hyphens), we encourage to encode information 
    into the IDs by adhering to a naming scheme. This will help keeping large 
    seqgra analyses with many different grammars and many different 
    architectures organized.

Grammar ID scheme
-----------------

.. note::
    Grammar IDs are defined by the root ID attribute of the data definition
    XML file, see :doc:`dd` for details. They are used as folder names in the
    the :doc:`seqgra output folder structure<structure>`.

.. raw:: html

    <p><code class="docutils literal notranslate"><span class="task-bold">[task]</span>-<span class="input-bold">[input-space]</span>-<span class="sim-exp-bold">[sim|exp]</span>-<span class="descriptor-bold">[grammar-descriptor]</span>-<span class="data-set-size-bold">[data-set-size]</span>-<span class="seed-bold">[simulation-seed]</span></code></p>

- :task-bold-border:`task`: :task-border:`mc` for multi-class classification, :task-border:`ml` for multi-label classification, followed by number of classes/labels
- :input-bold-border:`input-space`: :input-border:`dna` for DNA alphabet, :input-border:`protein` for protein alphabet, followed by width of input window
- :sim-exp-bold-border:`sim|exp`: :sim-exp-border:`sim` for simulated data, :sim-exp-border:`exp` for experimental data
- :descriptor-bold-border:`grammar-descriptor`: concisely describes grammar or experimental data
- :data-set-size-bold-border:`data-set-size`: total data set size (sum of training, validation, and test sets), usually using ``k`` for thousand, e.g., :data-set-size-border:`10k`, :data-set-size-border:`50k`, :data-set-size-border:`2000k`
- :seed-bold-border:`simulation-seed`: random seed used when simulating the data, always prefixed by ``s``, e.g., :seed-border:`s1`, :seed-border:`s5`, :seed-border:`s17`

**Examples:**

.. raw:: html

    <code class="docutils literal notranslate"><span class="task">mc2</span>-<span class="input">dna1000</span>-<span class="sim-exp">exp</span>-<span class="descriptor">sox2-oct4</span>-<span class="data-set-size">1000k</span>-<span class="seed">s1</span></code>:

- :task-border:`mc2`: multi-class classification task with 2 classes
- :input-border:`dna1000`: DNA input sequence space, 1000 nt input sequence window
- :sim-exp-border:`exp`: experimental data
- :descriptor-border:`sox2-oct4`: experimental data descriptor
- :data-set-size-border:`1000k`: data set contains 1,000,000 examples; sum of training, validation, and test sets
- :seed-border:`s1`: simulation seed 1

.. raw:: html

    <code class="docutils literal notranslate"><span class="task">ml50</span>-<span class="input">dna150</span>-<span class="sim-exp">sim</span>-<span class="descriptor">homer-interaction-order</span>-<span class="data-set-size">90k</span>-<span class="seed">s4</span></code>:

- :task-border:`ml50`: multi-label classification task with 50 labels
- :input-border:`dna150`: DNA input sequence space, 150 nt input sequence window
- :sim-exp-border:`sim`: simulated data
- :descriptor-border:`homer-interaction-order`: grammar descriptor
- :data-set-size-border:`90k`: data set contains 90,000 examples; sum of training, validation, and test sets
- :seed-border:`s4`: simulation seed 4

Model ID scheme
---------------

.. raw:: html

    <p><code class="docutils literal notranslate"><span class="library-bold">[library]</span>-<span class="task-bold">[task]</span>-<span class="input-bold">[input-space]</span>-<span class="descriptor-bold">[model-descriptor]</span>-<span class="seed-bold">[model-seed]</span></code></p>

- :library-bold-border:`library`: machine learning library the model is implemented in, either :library-border:`torch` for PyTorch, :library-border:`tf` for TensorFlow, or :library-border:`boc` for Bayes Optimal Classifier
- :task-bold-border:`task`: :task-border:`mc` for multi-class classification, :task-border:`ml` for multi-label classification, followed by number of classes/labels
- :input-bold-border:`input-space`: :input-border:`dna` for DNA alphabet, :input-border:`protein` for protein alphabet, followed by width of input window
- :descriptor-bold-border:`model-descriptor`: concisely describes model architecture, following its own scheme (see below)
- :seed-bold-border:`model-seed`: random seed used when training the model, always prefixed by ``s``, e.g., :seed-border:`s1`, :seed-border:`s5`, :seed-border:`s17`

**Examples:**

.. raw:: html

    <code class="docutils literal notranslate"><span class="library">torch</span>-<span class="task">ml2</span>-<span class="input">dna1000</span>-<span class="descriptor">conv10w-conv10w-gmp-fc5</span>-<span class="seed">s2</span></code>:

- :library-border:`torch`: model implemented using PyTorch library
- :task-border:`ml2`: multi-label classification task with 2 labels
- :input-border:`dna1000`: DNA input sequence space, 1000 nt input sequence window
- :descriptor-border:`conv10w-conv10w-gmp-fc5`: model descriptor, following its own scheme (see below)
- :seed-border:`s2`: model seed 2

.. raw:: html

    <code class="docutils literal notranslate"><span class="library">tf</span>-<span class="task">mc10</span>-<span class="input">dna150</span>-<span class="descriptor">conv10-do03-conv10-fc5-do03</span>-<span class="seed">s3</span></code>:

- :library-border:`tf`: model implemented using TensorFlow library
- :task-border:`mc10`: multi-class classification task with 10 labels
- :input-border:`dna150`: DNA input sequence space, 150 nt input sequence window
- :descriptor-border:`conv10-do03-conv10-fc5-do03`: model descriptor, following its own scheme (see below)
- :seed-border:`s3`: model seed 3

Model descriptor scheme
^^^^^^^^^^^^^^^^^^^^^^^

Model IDs contain a model descriptor, which is an attempt to provide as much information as possible about the architecture while being as concise as possible.
The output layer is never specified as it is determined by the classification task.

**General rules:**

- :descriptor-border:`conv`: convolutional layer
  
  - :descriptor-border:`conv10`: convolutional layer with 10 11-nt wide filters
  - :descriptor-border:`conv1xn`: convolutional layer with 1 3-nt wide filter
  - :descriptor-border:`conv5n`: convolutional layer with 5 5-nt wide filters
  - :descriptor-border:`conv50w`: convolutional layer with 50 21-nt wide filters
  - :descriptor-border:`conv2xw`: convolutional layer with 2 41-nt wide filters
  - :descriptor-border:`conv100xxw`: convolutional layer with 100 81-nt wide filters
- :descriptor-border:`fc`: dense or fully connected layer
  
  - :descriptor-border:`fc10`: fully connected layer with 10 units
- :descriptor-border:`gmp`: global max pooling operation
- :descriptor-border:`do`: dropout layer
  
  - :descriptor-border:`do03`: dropout layer with 30% dropout rate
- :descriptor-border:`bn`: batch normalization layer

**Examples:**

:descriptor-border:`conv10-do03-conv10-fc5-do03`: architecture with 

- :descriptor-border:`conv10`: convolutional layer with 10 11-nt wide filters,
- :descriptor-border:`do03`: dropout layer with 30% dropout rate,
- :descriptor-border:`conv10`: convolutional layer with 10 11-nt wide filters,
- :descriptor-border:`fc5`: fully connected layer with 5 units,
- :descriptor-border:`do03`: dropout layer with 30% dropout rate, 
- and output layer (always unspecified)

:descriptor-border:`conv10w-conv10w-gmp-fc5`: architecture with

- :descriptor-border:`conv10w`: convolutional layer with 10 21-nt wide filters,
- :descriptor-border:`conv10w`: convolutional layer with 10 21-nt wide filters,
- :descriptor-border:`gmp`: global max pooling operation,
- :descriptor-border:`fc5`: fully connected layer with 5 units,
- and output layer (always unspecified)

:descriptor-border:`deepsea`: known architecture, adjusted to fit classification task and input data
