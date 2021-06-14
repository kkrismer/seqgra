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
