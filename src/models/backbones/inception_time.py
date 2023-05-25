import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.wdcnn import ConvLayer


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,  # list
    ):
        super(InceptionModule, self).__init__()

        self.bottleneck = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv_s = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=10,
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv_m = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=20,
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv_l = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=40,
            stride=1,
            padding="same",
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(
            3, stride=1, padding=1
        )  # Padding depends on stride size
        self.conv_maxpool = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)

    def forward(self, x):
        z_bottleneck = self.bottleneck(x)
        z_maxpool = self.maxpool(x)

        z1 = self.conv_s(z_bottleneck)
        z2 = self.conv_m(z_bottleneck)
        z3 = self.conv_l(z_bottleneck)
        z4 = self.conv_maxpool(z_maxpool)

        z = torch.concatenate([z1, z2, z3, z4], dim=1)

        z = self.bn(z)
        z = F.relu(z)
        # z = F.hardswish(z) # TODO Try

        return z


# * Modified version of WDCNN for use as fewshot backbone
class InceptionTime_asd(nn.Module):
    def __init__(self, config):
        super(InceptionTime, self).__init__()

        self.config = config

        # Convolutional layers
        self.module_1 = InceptionModule(1, 32)
        self.module_2 = InceptionModule(32 * 4, 64)
        # self.module_3 = InceptionModule(32 * 4, 64)

        # Global average pooling
        self.globalAvgPool = nn.AvgPool1d(
            kernel_size=722
        )  # FIXME Kernel size (Depends on input length)

        # Classifier
        # self.fc1 = nn.Linear(
        #     64 * 4,
        #     self.config["embedding_len"],
        # )

        # # Optional FC layer weight initialization
        # if self.config["kaiming_init"]:
        #     self.apply(self._init_weights)

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

        if verbose:
            print(x.shape)

        # Conv layers

        out = self.module_1(x)
        if verbose:
            print(out.shape)

        out = self.module_2(out)
        if verbose:
            print(out.shape)

        # out = self.module_3(out)
        # if verbose:
        #     print(out.shape)

        # Global average pool
        out = self.globalAvgPool(out).squeeze()
        if verbose:
            print(out.shape)

        # Match to class num
        # out = self.fc1(out)
        # if verbose:
        #     print(out.shape)

        return out


class InceptionTime(nn.Module):
    def __init__(self, config):
        super(InceptionTime, self).__init__()
        self.config = config

        # Convolutional layers
        # FIXME Channel nums
        self.cn_layer1 = ConvLayer(
            1,  # * Multi sensor stuff is not part of the tests
            32,
            kernel_size=32,
            stride=8,
            padding=12,
            dropout=config["cl_dropout"],
        )
        self.cn_layer2 = ConvLayer(32, 64, dropout=config["fc_dropout"])
        self.cn_layer3 = ConvLayer(64, 128, dropout=config["fc_dropout"])
        # * Note the fc_dropout here
        # self.cn_layer4 = ConvLayer(64, 128, dropout=config["fc_dropout"])

        # Global average pooling
        self.globalAvgPool = nn.AvgPool1d(kernel_size=11)  # FIXME Kernel size

        # Classifier
        # self.fc1 = nn.Linear(
        #     128,
        #     config["embedding_len"],
        # )

        # Optional FC layer weight initialization
        # if self.config["kaiming_init"]:
        #     self.apply(self._init_weights)

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

        if verbose:
            print(x.shape)

        # Conv layers

        out = self.cn_layer1(x)
        if verbose:
            print(out.shape)

        out = self.cn_layer2(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer3(out)
        if verbose:
            print(out.shape)

        # out = self.cn_layer4(out)
        # if verbose:
        #     print(out.shape)

        # Global average pool
        out = self.globalAvgPool(out).squeeze()
        if verbose:
            print(out.shape)

        # Match channel num to embedding length
        # out = self.fc1(out)
        # if verbose:
        #     print(out.shape)

        return out
