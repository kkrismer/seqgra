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

        self.fc3 = torch.nn.Linear(in_features=1000, out_features=20)

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
