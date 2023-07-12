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
        self.cn_layer1 = nn.Sequential(
            # *2 because prototype and query embeddings are concatenated depth-wise
            nn.Conv1d(64 * 2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.cn_layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc1 = nn.Linear(64 * 64, 8)
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

        out = self.cn_layer1(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer2(out)
        if verbose:
            print(out.shape)

        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)
        out = F.relu(out)
        if verbose:
            print(out.shape)

        out = self.fc2(out)
        out = F.sigmoid(out)
        if verbose:
            print(out.shape)

        return out
