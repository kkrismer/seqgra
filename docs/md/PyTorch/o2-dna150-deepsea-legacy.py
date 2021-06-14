import math

import torch


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        INPUT_CHANNELS: int = 4
        CONV_FILTER_WIDTH: int = 8
        FC_NUM_UNITS: int = 925
        OUTPUT_UNITS: int = 2

        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(INPUT_CHANNELS,
                            320,
                            CONV_FILTER_WIDTH, 1,
                            math.floor(CONV_FILTER_WIDTH / 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(320,
                            480,
                            CONV_FILTER_WIDTH, 1,
                            math.floor(CONV_FILTER_WIDTH / 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(480,
                            960,
                            CONV_FILTER_WIDTH, 1,
                            math.floor(CONV_FILTER_WIDTH / 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.Dropout(0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(1920, FC_NUM_UNITS),
            torch.nn.ReLU(),
            torch.nn.Linear(FC_NUM_UNITS, OUTPUT_UNITS)
        )

    def forward(self, x):
        x = self.model(x)
        return x
