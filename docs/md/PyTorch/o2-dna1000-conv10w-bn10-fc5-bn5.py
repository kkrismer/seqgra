import math

import torch


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        INPUT_CHANNELS: int = 4
        INPUT_WIDTH: int = 1000
        CONV_NUM_FILTERS: int = 10
        CONV_FILTER_WIDTH: int = 21
        FC_NUM_UNITS: int = 5
        OUTPUT_UNITS: int = 2

        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(INPUT_CHANNELS,
                            CONV_NUM_FILTERS,
                            CONV_FILTER_WIDTH, 1,
                            math.floor(CONV_FILTER_WIDTH / 2)),
            torch.nn.BatchNorm1d(num_features=CONV_NUM_FILTERS),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(INPUT_WIDTH * CONV_NUM_FILTERS, FC_NUM_UNITS),
            torch.nn.BatchNorm1d(num_features=FC_NUM_UNITS),
            torch.nn.ReLU(),
            torch.nn.Linear(FC_NUM_UNITS, OUTPUT_UNITS)
        )

    def forward(self, x):
        x = self.model(x)
        return x
