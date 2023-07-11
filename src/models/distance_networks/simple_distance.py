import math

import torch.nn as nn
import torch.nn.functional as F


class SimpleDistanceNetwork(nn.Module):
    def __init__(self, config):
        super(SimpleDistanceNetwork, self).__init__()
        self.config = config

        # self.conv = nn.Conv1d(
        #     2,
        #     1,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=False,
        # )
        self.conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(1, momentum=1, affine=True),
            nn.ReLU(),
            # nn.MaxPool1d(2),
        )

        self.fc1 = nn.Linear(128, 8)
        # self.fc1 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(8, 1)

        # Optional FC layer weight initialization
        if self.config["kaiming_init"]:
            self.apply(self._init_weights)

    # For Kaiming weight initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x):
        verbose = False

        out = x

        if verbose:
            print("Input:", x.shape)

        out = self.conv(out)
        # out = F.relu(out)
        out = out.squeeze()
        if verbose:
            print(out.shape)

        out = self.fc1(out)
        out = F.relu(out)
        # out = F.sigmoid(out)
        if verbose:
            print(out.shape)

        out = self.fc2(out)
        out = F.sigmoid(out)
        if verbose:
            print(out.shape)

        return out
