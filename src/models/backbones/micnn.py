import math

import torch.nn as nn

from models.backbones.wdcnn import ConvLayer


class MiCNN(nn.Module):
    def __init__(self, config):
        super(MiCNN, self).__init__()
        self.config = config

        # Convolutional layers
        # FIXME Channel nums
        self.cn_layer1 = ConvLayer(
            1,  # * Multi sensor stuff is not part of the tests
            16,
            kernel_size=64
            if "FFT" not in self.config["preprocessing_batch"]
            else 32,  # FIXME when better idea of specific FFT implementation
            stride=16 if "FFT" not in self.config["preprocessing_batch"] else 8,
            padding=24 if "FFT" not in self.config["preprocessing_batch"] else 16,
            dropout=config["cl_dropout"],
        )
        self.cn_layer2 = ConvLayer(16, 32, dropout=config["cl_dropout"])
        self.cn_layer3 = ConvLayer(32, 64, dropout=config["cl_dropout"])
        # * Note the fc_dropout here
        self.cn_layer4 = ConvLayer(64, 64, dropout=config["fc_dropout"])

        # Global average pooling
        self.globalAvgPool = nn.AvgPool1d(kernel_size=12)  # FIXME Kernel size

        # Classifier
        self.fc1 = nn.Linear(
            64,  # FIXME to match channel size
            config["embedding_len"],
        )

        # Optional FC layer weight initialization
        if self.config["kaiming_init"]:
            self.apply(self._init_weights)

    # For Kaiming weight initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(
                module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    module.weight)
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

        out = self.cn_layer4(out)
        if verbose:
            print(out.shape)

        # Global average pool
        out = self.globalAvgPool(out).squeeze()
        if verbose:
            print(out.shape)

        # Match channel num to embedding length
        out = self.fc1(out)
        if verbose:
            print(out.shape)

        return out
