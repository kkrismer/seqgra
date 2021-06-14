import math

import torch


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        INPUT_CHANNELS: int = 4
        CONV_NUM_FILTERS: int = 1
        CONV_FILTER_WIDTH: int = 3
        OUTPUT_UNITS: int = 2

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(INPUT_CHANNELS,
                            CONV_NUM_FILTERS,
                            CONV_FILTER_WIDTH, 1,
                            math.floor(CONV_FILTER_WIDTH / 2)),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(CONV_NUM_FILTERS, OUTPUT_UNITS)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
