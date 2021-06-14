import math

import torch


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        INPUT_CHANNELS: int = 4
        CONV_FILTER_WIDTH: int = 8
        FC_NUM_UNITS: int = 925
        OUTPUT_UNITS: int = 100

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
            torch.nn.Linear(14400, FC_NUM_UNITS),
            torch.nn.ReLU(),
            torch.nn.Linear(FC_NUM_UNITS, OUTPUT_UNITS)
        )

        # self.conv1 = torch.nn.Conv1d(INPUT_CHANNELS,
        #                     320,
        #                     CONV_FILTER_WIDTH, 1,
        #                     math.floor(CONV_FILTER_WIDTH / 2))
        # self.conv2 = torch.nn.Conv1d(320,
        #                     480,
        #                     CONV_FILTER_WIDTH, 1,
        #                     math.floor(CONV_FILTER_WIDTH / 2))
        # self.conv3 = torch.nn.Conv1d(480,
        #                     960,
        #                     CONV_FILTER_WIDTH, 1,
        #                     math.floor(CONV_FILTER_WIDTH / 2))
        # self.fc1 = torch.nn.Linear(14400, FC_NUM_UNITS)
        # self.fc2 = torch.nn.Linear(FC_NUM_UNITS, OUTPUT_UNITS)

    def forward(self, x):
        x = self.model(x)
        return x

        # x = self.conv1(x)
        # x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.max_pool1d(x, 4)
        # x = torch.nn.functional.dropout(x, 0.2)

        # x = self.conv2(x)
        # x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.max_pool1d(x, 4)
        # x = torch.nn.functional.dropout(x, 0.2)

        # x = self.conv3(x)
        # x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.max_pool1d(x, 4)
        # x = torch.nn.functional.dropout(x, 0.5)

        # # print(self.num_flat_features(x))
        # x = x.view(-1, self.num_flat_features(x))

        # x = self.fc1(x)
        # x = torch.nn.functional.relu(x)

        # x = self.fc2(x)
        # return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
