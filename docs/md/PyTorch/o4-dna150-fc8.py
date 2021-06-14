import torch


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        INPUT_CHANNELS: int = 4
        INPUT_WIDTH: int = 150
        FC_NUM_UNITS: int = 8
        OUTPUT_UNITS: int = 4

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(INPUT_WIDTH * INPUT_CHANNELS, FC_NUM_UNITS),
            torch.nn.ReLU(),
            torch.nn.Linear(FC_NUM_UNITS, OUTPUT_UNITS)
        )

    def forward(self, x):
        x = self.model(x)
        return x
