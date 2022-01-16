Model definitions of common deep learning architectures
=======================================================

This page contains model definitions for deep learning architectures 
that are used by three popular methods in computational genomics, Basset, 
ChromDragoNN, and DeepSEA.

For an in-depth description of the model definition language, see :doc:`md`.

The source code of the architectures shown here (and many more) is also 
`available on GitHub <https://github.com/kkrismer/seqgra/tree/master/docsrc/defs/md>`_.

.. note::
    The architectures presented here might deviate from how they were presented
    in their respective research articles. (1) The input layers were adapted to 
    take one-hot encoded 1000 bp DNA sequences. (2) The output layers were 
    adapted for 50 classes. (3) The loss was adapted for multi-class
    classification.

Basset
------

Source: `DOI: 10.1101/gr.200535.115 <https://doi.org/10.1101/gr.200535.115>`_

**Model definition:**

`torch-mc50-dna1000-basset-s1.xml`:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>

    <seqgramodel xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="https://kkrismer.github.io/seqgra/model-config.xsd">
        <general id="torch-mc50-dna1000-basset-s1">
            <name>Basset architecture</name>
            <description>1000 bp DNA sequence window, 3 convolutional layers, 2 fully connected layer</description>
            <task>multi-class classification</task>
            <sequencespace>DNA</sequencespace>
            <library>PyTorch</library>
            <inputencoding>1D</inputencoding>
            <labels>
                <pattern prefix="c" postfix="" min="1" max="50"/>
            </labels>
            <seed>1</seed>
        </general>
        <architecture>
            <external format="pytorch-module" classname="TorchModel">PyTorch/o50-dna1000-basset.py</external>
        </architecture>
        <loss>
            <hyperparameter name="loss">CrossEntropyLoss</hyperparameter>
        </loss>
        <optimizer>
            <hyperparameter name="optimizer">Adam</hyperparameter>
            <hyperparameter name="learning_rate">0.0001</hyperparameter>
            <hyperparameter name="clipnorm">0.5</hyperparameter>
        </optimizer>
        <trainingprocess>
            <hyperparameter name="batch_size">100</hyperparameter>
            <hyperparameter name="epochs">100</hyperparameter>
            <hyperparameter name="early_stopping">True</hyperparameter>
            <hyperparameter name="shuffle">True</hyperparameter>
        </trainingprocess>
    </seqgramodel>

**Python implementation:**

`PyTorch/o50-dna1000-basset.py`:

.. code-block:: python

    # adapted from https://github.com/withai/PyBasset/blob/master/pytorch_script.py

    import torch


    class TorchModel(torch.nn.Module):
        def __init__(self, ):
            super().__init__()
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=4, out_channels=300, kernel_size=19),
                torch.nn.BatchNorm1d(num_features=300),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=3))

            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=300, out_channels=200, kernel_size=11),
                torch.nn.BatchNorm1d(num_features=200),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=4))

            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=200, out_channels=200, kernel_size=7),
                torch.nn.BatchNorm1d(num_features=200),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=4))

            self.fc1 = torch.nn.Linear(in_features=3600, out_features=1000)
            self.relu4 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(p=0.3)

            self.fc2 = torch.nn.Linear(in_features=1000, out_features=1000)
            self.relu5 = torch.nn.ReLU()
            self.dropout2 = torch.nn.Dropout(p=0.3)

            self.fc3 = torch.nn.Linear(in_features=1000, out_features=50)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu4(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu5(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

ChromDragoNN
------------

Source: `DOI: 10.1093/bioinformatics/btz352 <https://doi.org/10.1093/bioinformatics/btz352>`_

**Model definition:**

`torch-mc50-dna1000-chromdragonn-s1.xml`:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>

    <seqgramodel xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="https://kkrismer.github.io/seqgra/model-config.xsd">
        <general id="torch-mc50-dna1000-chromdragonn-s1">
            <name>ChromDragoNN architecture</name>
            <description>1000 bp DNA sequence window, residual convolutional network</description>
            <task>multi-class classification</task>
            <sequencespace>DNA</sequencespace>
            <library>PyTorch</library>
            <inputencoding>1D</inputencoding>
            <labels>
                <pattern prefix="c" postfix="" min="1" max="50"/>
            </labels>
            <seed>1</seed>
        </general>
        <architecture>
            <external format="pytorch-module" classname="TorchModel">PyTorch/o50-dna1000-chromdragonn.py</external>
        </architecture>
        <loss>
            <hyperparameter name="loss">CrossEntropyLoss</hyperparameter>
        </loss>
        <optimizer>
            <hyperparameter name="optimizer">Adam</hyperparameter>
            <hyperparameter name="learning_rate">0.002</hyperparameter>
        </optimizer>
        <trainingprocess>
            <hyperparameter name="batch_size">256</hyperparameter>
            <hyperparameter name="epochs">100</hyperparameter>
            <hyperparameter name="early_stopping">True</hyperparameter>
            <hyperparameter name="shuffle">True</hyperparameter>
        </trainingprocess>
    </seqgramodel>

**Python implementation:**

`PyTorch/o50-dna1000-chromdragonn.py`:

.. code-block:: python

    # adapted from https://github.com/kundajelab/ChromDragoNN/blob/master/model_zoo/stage1/resnet.pychromdragonn

    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class L1Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
            self.bn2 = nn.BatchNorm2d(64)
            self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(
                inplace=True), self.conv2, self.bn2)

        def forward(self, x):
            out = self.layer(x)
            out += x
            out = F.relu(out)
            return out


    class L2Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
            self.conv2 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(128)
            self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(
                inplace=True), self.conv2, self.bn2)

        def forward(self, x):
            out = self.layer(x)
            out += x
            out = F.relu(out)
            return out


    class L3Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
            self.conv2 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))
            self.conv3 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))

            self.bn1 = nn.BatchNorm2d(200)
            self.bn2 = nn.BatchNorm2d(200)
            self.bn3 = nn.BatchNorm2d(200)

            self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                    self.conv2, self.bn2, nn.ReLU(inplace=True),
                                    self.conv3, self.bn3)

        def forward(self, x):
            out = self.layer(x)
            out += x
            out = F.relu(out)
            return out


    class L4Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
            self.bn1 = nn.BatchNorm2d(200)
            self.conv2 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
            self.bn2 = nn.BatchNorm2d(200)
            self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                    self.conv2, self.bn2)

        def forward(self, x):
            out = self.layer(x)
            out += x
            out = F.relu(out)
            return out


    class TorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = 0.3
            self.num_cell_types = 50
            self.blocks = [2, 2, 2, 2]

            self.conv1 = nn.Conv2d(4, 48, (3, 1), stride=(1, 1), padding=(1, 0))
            self.bn1 = nn.BatchNorm2d(48)
            self.conv2 = nn.Conv2d(48, 64, (3, 1), stride=(1, 1), padding=(1, 0))
            self.bn2 = nn.BatchNorm2d(64)
            self.prelayer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True),
                                        self.conv2, self.bn2, nn.ReLU(inplace=True))

            self.layer1 = nn.Sequential(*[L1Block()
                                        for x in range(self.blocks[0])])
            self.layer2 = nn.Sequential(*[L2Block()
                                        for x in range(self.blocks[1])])
            self.layer3 = nn.Sequential(*[L3Block()
                                        for x in range(self.blocks[2])])
            self.layer4 = nn.Sequential(*[L4Block()
                                        for x in range(self.blocks[3])])

            self.c1to2 = nn.Conv2d(64, 128, (3, 1), stride=(1, 1), padding=(1, 0))
            self.b1to2 = nn.BatchNorm2d(128)
            self.l1tol2 = nn.Sequential(
                self.c1to2, self.b1to2, nn.ReLU(inplace=True))

            self.c2to3 = nn.Conv2d(128, 200, (1, 1), padding=(3, 0))
            self.b2to3 = nn.BatchNorm2d(200)
            self.l2tol3 = nn.Sequential(
                self.c2to3, self.b2to3, nn.ReLU(inplace=True))

            self.maxpool1 = nn.MaxPool2d((3, 1))
            self.maxpool2 = nn.MaxPool2d((4, 1))
            self.maxpool3 = nn.MaxPool2d((4, 1))
            self.fc1 = nn.Linear(4200, 1000)
            self.bn4 = nn.BatchNorm1d(1000)
            self.fc2 = nn.Linear(1000, 1000)
            self.bn5 = nn.BatchNorm1d(1000)
            self.fc3 = nn.Linear(1000, self.num_cell_types)
            self.flayer = self.final_layer()

        def final_layer(self):
            self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))
            self.bn3 = nn.BatchNorm2d(200)
            return nn.Sequential(self.conv3, self.bn3, nn.ReLU(inplace=True))

        def forward(self, s):
            s = s.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
            s = s.view(-1, 4, 1000, 1)  # batch_size x 4 x 1000 x 1 [4 channels]

            out = self.prelayer(s)
            out = self.layer1(out)
            out = self.layer2(self.l1tol2(out))
            out = self.maxpool1(out)
            out = self.layer3(self.l2tol3(out))
            out = self.maxpool2(out)
            out = self.layer4(out)
            out = self.flayer(out)
            out = self.maxpool3(out)
            out = out.view(-1, 4200)
            conv_out = out
            out = F.dropout(F.relu(self.bn4(self.fc1(out))), p=self.dropout,
                            training=self.training)  # batch_size x 1000
            out = F.dropout(F.relu(self.bn5(self.fc2(out))), p=self.dropout,
                            training=self.training)  # batch_size x 1000
            out = self.fc3(out)
            return out

DeepSEA
-------

Source: `DOI: 10.1038/nmeth.3547 <https://doi.org/10.1038/nmeth.3547>`_

**Model definition:**

`torch-mc50-dna1000-deepsea-s1.xml`:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>

    <seqgramodel xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="https://kkrismer.github.io/seqgra/model-config.xsd">
        <general id="torch-mc50-dna1000-deepsea-s1">
            <name>DeepSEA architecture</name>
            <description>1000 bp DNA sequence window, 3 convolutional layers, 1 fully connected layer</description>
            <task>multi-class classification</task>
            <sequencespace>DNA</sequencespace>
            <library>PyTorch</library>
            <inputencoding>1D</inputencoding>
            <labels>
                <pattern prefix="c" postfix="" min="1" max="50"/>
            </labels>
            <seed>1</seed>
        </general>
        <architecture>
            <external format="pytorch-module" classname="TorchModel">PyTorch/o50-dna1000-deepsea.py</external>
        </architecture>
        <loss>
            <hyperparameter name="loss">CrossEntropyLoss</hyperparameter>
        </loss>
        <optimizer>
            <hyperparameter name="optimizer">SGD</hyperparameter>
            <hyperparameter name="learning_rate">0.01</hyperparameter>
            <hyperparameter name="momentum">0.9</hyperparameter>
        </optimizer>
        <trainingprocess>
            <hyperparameter name="batch_size">100</hyperparameter>
            <hyperparameter name="epochs">100</hyperparameter>
            <hyperparameter name="early_stopping">True</hyperparameter>
            <hyperparameter name="shuffle">True</hyperparameter>
        </trainingprocess>
    </seqgramodel>

**Python implementation:**

`PyTorch/o50-dna1000-deepsea.py`:

.. code-block:: python

    # adapted from https://github.com/PuYuQian/PyDeepSEA/blob/master/DeepSEA_train.py

    import torch


    class TorchModel(torch.nn.Module):
        def __init__(self, ):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(
                in_channels=4, out_channels=320, kernel_size=8)
            self.conv2 = torch.nn.Conv1d(
                in_channels=320, out_channels=480, kernel_size=8)
            self.conv3 = torch.nn.Conv1d(
                in_channels=480, out_channels=960, kernel_size=8)
            self.maxpool = torch.nn.MaxPool1d(kernel_size=4, stride=4)
            self.drop1 = torch.nn.Dropout(p=0.2)
            self.drop2 = torch.nn.Dropout(p=0.5)
            self.linear1 = torch.nn.Linear(53 * 960, 925)
            self.linear2 = torch.nn.Linear(925, 50)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.nn.functional.relu(x)
            x = self.maxpool(x)
            x = self.drop1(x)
            x = self.conv2(x)
            x = torch.nn.functional.relu(x)
            x = self.maxpool(x)
            x = self.drop1(x)
            x = self.conv3(x)
            x = torch.nn.functional.relu(x)
            x = self.drop2(x)
            x = x.view(-1, 53 * 960)
            x = self.linear1(x)
            x = torch.nn.functional.relu(x)
            x = self.linear2(x)
            return x
