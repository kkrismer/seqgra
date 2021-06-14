# adapted from https://github.com/PuYuQian/PyDanQ/blob/master/DanQ_train.py

import torch


class TorchModel(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=4, out_channels=320,
                                     kernel_size=26)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=13, stride=13)
        self.drop1 = torch.nn.Dropout(p=0.2)
        self.bilstm = torch.nn.LSTM(input_size=320, hidden_size=320,
                                    num_layers=2,
                                    batch_first=True,
                                    dropout=0.5,
                                    bidirectional=True)
        self.linear1 = torch.nn.Linear(75 * 640, 925)
        self.linear2 = torch.nn.Linear(925, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool(x)
        x = self.drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n, h_c) = self.bilstm(x_x)
        x = x.contiguous().view(-1, 75 * 640)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x
