import math

import torch.nn as nn
import torch.nn.functional as F


class VerySimpleDistanceNetwork(nn.Module):
    def __init__(self, config):
        super(VerySimpleDistanceNetwork, self).__init__()
        self.config = config
        self.embedding_channels = 64
        self.hidden_size = 16

        self.fc2 = nn.Linear(self.embedding_channels * 2, 1)
        # self.fc1 = nn.Linear(self.embedding_channels * 2, self.hidden_size)
        # self.fc2 = nn.Linear(self.hidden_size, 1)

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

        out = out.reshape(out.shape[0], -1)

        # out = self.fc1(out)
        # out = F.hardswish(out)
        # # out = F.relu(out)
        # if verbose:
        #     print("FC 1:", out.shape)

        out = self.fc2(out)
        out = F.sigmoid(out)
        if verbose:
            print("FC 2:", out.shape)
            quit()

        return out
