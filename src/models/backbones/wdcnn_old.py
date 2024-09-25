import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.pool(out)

        return out


class WDCNN_old(nn.Module):
    def __init__(self, config, bias=False):
        super(WDCNN_old, self).__init__()

        self.config = config

        self.cn_layer1 = ConvLayer(1, 16, kernel_size=64, stride=16, padding=24, bias=bias)
        self.cn_layer2 = ConvLayer(16, 32, bias=bias)
        self.cn_layer3 = ConvLayer(32, 64, bias=bias)
        self.cn_layer4 = ConvLayer(64, 64, bias=bias)
        self.cn_layer5 = ConvLayer(64, 64, padding=0, bias=bias)

        # classifier
        self.fc1 = nn.Linear(
            256 if self.config["FFT"] else 640,
            self.config["embedding_len"],
        )

        if self.config["kaiming_init"]:
            self.apply(self._init_weights)

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

        out = self.cn_layer1(x)
        if verbose:
            print(out.shape)

        out = self.cn_layer2(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer3(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer4(out)
        if verbose:
            print(out.shape)

        out = self.cn_layer5(out)
        if verbose:
            print(out.shape)

        out = out.view(out.shape[0], -1)
        if verbose:
            print(out.shape)

        out = self.fc1(out)
        out = F.relu(out)
        if verbose:
            print(out.shape)

        # Normalize embedding to unit length
        # if self.config.embedding_unit_len:
        if self.config["embedding_unit_len"]:
            out = F.normalize(out, p=1.0, dim=1) * 1000

        return out
