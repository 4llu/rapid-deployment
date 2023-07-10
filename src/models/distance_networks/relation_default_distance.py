import torch.nn as nn
import torch.nn.functional as F

from models.backbones.wdcnn import ConvLayer

class DefaultDistanceNetwork(nn.Module):
    def __init__(self, config):
        super(DefaultDistanceNetwork, self).__init__()
        self.config = config

        self.cn_layer1 = ConvLayer(
            1, 1,
            # 64, 64,
            kernel_size=3,
            stride=1,
            padding="same",
            dropout=config["cl_dropout"],
        )
        self.cn_layer2 = ConvLayer(
            1, 1,
            # 64, 8,
            kernel_size=3,
            stride=1,
            padding="same",
            dropout=config["cl_dropout"],
        )

        self.fc1 = nn.Linear(128*1, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        verbose = False

        if verbose:
            print("Input:", x.shape)

        # Conv layers

        out = self.cn_layer1(x)
        if verbose:
            print("CL 1:", out.shape)

        out = self.cn_layer2(out)
        if verbose:
            print("CL 2:", out.shape)

        # Flatten channels
        out = out.view(out.shape[0], -1)
        if verbose:
            print(out.shape)

        out = self.fc1(out)
        out = F.relu(out)
        if verbose:
            print(out.shape)

        out = self.fc2(out)
        out = F.sigmoid(out)
        if verbose:
            print(out.shape)

        return out