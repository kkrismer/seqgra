Model definition
================

The seqgra model definition language is an XML-based specification of the 
model creation and training process. Each model definition XML file contains a 
complete description for the seqgra learner to create and train a neural 
network architecture.

.. image:: _static/md.svg
    :width: 50%
    :alt: seqgra model definition schematic

    
The language is defined using XML schema 
(`model definition XML schema file <https://kkrismer.github.io/seqgra/model-config.xsd>`_).

This document describes the sections of a valid seqgra model definition. 
For examples of model definitions, see the 
`model definitions folder on GitHub <https://github.com/kkrismer/seqgra/tree/master/docsrc/defs/md>`_ 
and the section about common model architectures 
(see :doc:`common-architectures`), including Basset, ChromDragoNN, and DeepSEA.

General information
-------------------

This section contains meta-info about the model.

**Example 1 - PyTorch binary classification model on 150bp DNA sequence window:**

.. code-block:: xml

    <general id="torch-mc2-dna150-conv1-gmp-fc5-s1">
        <name>1 conv layer with 1 11-nt wide filters, global max pooling, 1 fully connected layer with 5 units</name>
        <description></description>
        <task>multi-class classification</task>
        <sequencespace>DNA</sequencespace>
        <library>PyTorch</library>
        <inputencoding>1D</inputencoding>
        <labels>
            <label>c1</label>
            <label>c2</label>
        </labels>
        <seed>1</seed>
    </general>

**Example 2 - TensorFlow multi-label classification model on 1000bp DNA sequence window:**

.. code-block:: xml

    <general id="tf-ml10-dna1000-conv10w-conv10w-gmp-fc10-s1">
        <name>conv layer with 10 21-nt wide filters, conv layer with 10 21-nt wide filters, global max pooling, fully connected layer with 10 units</name>
        <description></description>
        <task>multi-label classification</task>
        <sequencespace>DNA</sequencespace>
        <library>TensorFlow</library>
        <labels>
            <pattern prefix="c" postfix="" min="1" max="10"/>
        </labels>
        <seed>1</seed>
    </general>

**Example 3 - Bayes Optimal Classifier binary classification model on 1000bp DNA sequence window:**

.. code-block:: xml

    <general id="boc-mc2-dna1000-homer-s1">
        <name>Bayes optimal classifier for mc2-dna1000-homer</name>
        <description>using the same PWMs that were used to generate the data</description>
        <task>multi-class classification</task>
        <sequencespace>DNA</sequencespace>
        <library>BayesOptimalClassifier</library>
        <labels>
            <pattern prefix="c" postfix="" min="1" max="2"/>
        </labels>
        <seed>1</seed>
    </general>

- a valid model ID can only contain ``[A-Za-z0-9_-]+``
- model name and description can be left empty
- task can be either *multi-class classification* or *multi-label classification*
- sequence space can be one of the following: *DNA*, *RNA*, *protein*
- library can be one of the following: *PyTorch*, *TensorFlow*, *BayesOptimalClassifier*
- seed is the model seed, which affects weight initialization and SGD (integer)
  
Architecture
------------

Defines the neural network architecture. For PyTorch, architecture is defined 
externally with ``torch.nn.Module`` derived class. For TensorFlow, 
tf.keras.Sequential architectures can be embedded in XML, while external 
architectures can be loaded for more expressivity. For Bayes Optimal 
Classifier, architecture 
is defined by reference to a data definition file.

**Example 1 - PyTorch:**

.. code-block:: xml

    <architecture>
        <external format="pytorch-module" classname="TorchModel">PyTorch/o2-dna150-conv10-do03-conv10-fc5-do03.py</external>
    </architecture>

.. code-block:: python

    import math

    import torch

    class TorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            INPUT_CHANNELS: int = 4
            CONV1_NUM_FILTERS: int = 10
            CONV2_NUM_FILTERS: int = 10
            CONV_FILTER_WIDTH: int = 11
            FC_NUM_UNITS: int = 5
            OUTPUT_UNITS: int = 2

            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(INPUT_CHANNELS,
                                CONV1_NUM_FILTERS,
                                CONV_FILTER_WIDTH, 1,
                                math.floor(CONV_FILTER_WIDTH / 2)),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.3),
                torch.nn.Conv1d(CONV1_NUM_FILTERS,
                                CONV2_NUM_FILTERS,
                                CONV_FILTER_WIDTH, 1,
                                math.floor(CONV_FILTER_WIDTH / 2)),
                torch.nn.ReLU(),
                torch.nn.AdaptiveMaxPool1d(1)
            )

            self.fc = torch.nn.Sequential(
                torch.nn.Linear(CONV2_NUM_FILTERS, FC_NUM_UNITS),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(FC_NUM_UNITS, OUTPUT_UNITS)
            )

        def forward(self, x):
            batch_size = x.size(0)
            x = self.conv(x)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            return x

**Example 2 - TensorFlow (embedded, tf.keras.Sequential model):**

.. code-block:: xml

    <architecture>
        <sequential>
            <operation input_shape="(1000, 4)" kernel_size="21" filters="10" activation="relu">Conv1D</operation>
            <operation kernel_size="21" filters="10" activation="relu">Conv1D</operation>
            <operation>GlobalMaxPool1D</operation>
            <operation units="10" activation="relu">Dense</operation>
            <operation units="10" activation="softmax">Dense</operation>
        </sequential>
    </architecture>

**Example 3 - TensorFlow (external, arbitrary model, SavedModel format):**

.. code-block:: xml

    <architecture>
        <external format="keras-tf-whole-model">TensorFlow/basic-model</external>
    </architecture>

.. code-block:: python

    import tensorflow as tf
    
    inputs = tf.keras.Input(shape=(32,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.save("TensorFlow/basic-model")

**Example 4 - TensorFlow (external, arbitrary model, H5 format):**

.. code-block:: xml

    <architecture>
        <external format="keras-h5-whole-model">TensorFlow/basic-model.h5</external>
    </architecture>

.. code-block:: python

    import tensorflow as tf
    
    inputs = tf.keras.Input(shape=(32,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")

    test_input = np.random.random((128, 32))
    test_target = np.random.random((128, 1))
    model.fit(test_input, test_target)

    model.save("TensorFlow/basic-model.h5", save_format="h5")

**Example 5 - Bayes Optimal Classifier:**

.. code-block:: xml

    <architecture>
        <external format="data-definition">defs/data/mc2-dna1000-homer-interaction-spacing-100k-s1.xml</external>
    </architecture>

Loss
----

This section defines hyperparameters pertaining to the loss function.

**Example 1 - PyTorch multi-class classification:**

.. code-block:: xml

    <loss>
        <hyperparameter name="loss">CrossEntropyLoss</hyperparameter>
    </loss>

**Example 2 - PyTorch multi-label classification:**

.. code-block:: xml

    <loss>
        <hyperparameter name="loss">BCEWithLogitsLoss</hyperparameter>
    </loss>

**Example 3 - TensorFlow multi-class classification:**

.. code-block:: xml

    <loss>
        <hyperparameter name="loss">categorical_crossentropy</hyperparameter>
    </loss>

**Example 4 - TensorFlow multi-label classification:**

.. code-block:: xml

    <loss>
        <hyperparameter name="loss">binary_crossentropy</hyperparameter>
    </loss>

Optimizer
---------

This section defines hyperparameters pertaining to the optimizer.

**Example 1 - PyTorch with Adam:**

.. code-block:: xml

    <optimizer>
        <hyperparameter name="optimizer">Adam</hyperparameter>
        <hyperparameter name="learning_rate">0.0001</hyperparameter>
        <hyperparameter name="clipnorm">0.5</hyperparameter>
    </optimizer>
    
**Example 2 - TensorFlow with SGD:**

.. code-block:: xml

    <optimizer>
        <hyperparameter name="optimizer">SGD</hyperparameter>
        <hyperparameter name="learning_rate">0.001</hyperparameter>
        <hyperparameter name="momentum">0.9</hyperparameter>
    </optimizer>

Training process
----------------

This section defines hyperparameters pertaining to the training process.

**Example:**

.. code-block:: xml

    <trainingprocess>
        <hyperparameter name="batch_size">100</hyperparameter>
        <hyperparameter name="epochs">100</hyperparameter>
        <hyperparameter name="early_stopping">True</hyperparameter>
        <hyperparameter name="shuffle">True</hyperparameter>
    </trainingprocess>
