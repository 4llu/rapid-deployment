import torch.nn as nn

from models.backbones.wdcnn import ConvLayer


class RelationDefault(nn.Module):
    def __init__(self, config):
        super(RelationDefault, self).__init__()
        self.config = config

        # Convolutional layers
        # FIXME Channel nums
        self.cn_layer1 = ConvLayer(
            1, 16,
            kernel_size=3,
            stride=1,
            padding="same",
            dropout=config["cl_dropout"],
        )
        self.cn_layer2 = ConvLayer(
            16, 32,
            kernel_size=3,
            stride=1,
            padding="same",
            dropout=config["cl_dropout"],
        )
        self.cn_layer3 = ConvLayer(
            32, 64,
            kernel_size=3,
            stride=1,
            padding="same",
            dropout=config["cl_dropout"],
        )
        self.cn_layer4 = ConvLayer(
            64, 64,
            kernel_size=3,
            stride=1,
            padding="same",
            dropout=config["cl_dropout"],
        )

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

        out = self.cn_layer3(out)
        if verbose:
            print("CL 3:", out.shape)

        out = self.cn_layer4(out)
        if verbose:
            print("CL 4:", out.shape)

        return out
