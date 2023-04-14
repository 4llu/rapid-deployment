import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        dropout=0.0,
    ):
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
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.pool(out)

        return out


class WDCNN(nn.Module):
    def __init__(self, input_channels, n_classes, config, bias=False):
        super(WDCNN, self).__init__()
        self.config = config

        self.cn_layer1 = ConvLayer(
            input_channels,
            16,
            kernel_size=64 if "FFT" not in self.config["preprocessing"] else 32,
            stride=16 if "FFT" not in self.config["preprocessing"] else 8,
            padding=24 if "FFT" not in self.config["preprocessing"] else 16,
            bias=bias,
            dropout=config["cl_dropout"],
        )
        self.cn_layer2 = ConvLayer(16, 32, bias=bias, dropout=config["cl_dropout"])
        self.cn_layer3 = ConvLayer(32, 64, bias=bias, dropout=config["cl_dropout"])
        self.cn_layer4 = ConvLayer(64, 64, bias=bias, dropout=config["cl_dropout"])
        self.cn_layer5 = ConvLayer(
            64, 64, padding=0, bias=bias, dropout=config["cl_dropout"]
        )

        # Classifier
        self.fc1 = nn.Linear(
            128,
            config["fc1_output_size"],
            # 256 if self.config["FFT"] else 640,
            # 18,
        )
        # self.bn1 = nn.BatchNorm1d(config["wdcnn_fc1_output_size"])
        self.fc2 = nn.Linear(config["wdcnn_fc1_output_size"], n_classes)

        self.dropout_fc = nn.Dropout1d(p=config["fc_dropout"])

    def forward(self, x):
        # verbose = True

        # if verbose:
        #     print(x.shape)

        out = self.cn_layer1(x)
        # if verbose:
        #     print(out.shape)

        out = self.cn_layer2(out)
        # if verbose:
        #     print(out.shape)

        out = self.cn_layer3(out)
        # if verbose:
        #     print(out.shape)

        out = self.cn_layer4(out)
        # if verbose:
        #     print(out.shape)

        out = self.cn_layer5(out)
        # if verbose:
        #     print(out.shape)

        # Reshape channels
        n_features = out.shape[1] * out.shape[2]
        out = out.view(-1, n_features).contiguous()
        # if verbose:
        #     print(out.shape)

        out = F.relu(self.fc1(out))
        out = self.dropout_fc(out)
        # out = self.bn1(out)
        # if verbose:
        #     print(out.shape)

        out = self.fc2(out)

        return out
