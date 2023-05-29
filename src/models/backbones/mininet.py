import math

import torch.nn as nn


class mininet(nn.Module):
    def __init__(self, config):
        super(mininet, self).__init__()
        self.config = config

        # Classifier
        self.fc1 = nn.Linear(
            # 722,
            3000,
            config["embedding_len"],
            # 256,
        )

        # self.fc2 = nn.Linear(
        #     256,
        #     config["embedding_len"],
        # )

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

        out = self.fc1(x)
        if verbose:
            print(out.shape)

        # out = self.fc2(out)
        # if verbose:
        #     print(out.shape)

        return out
