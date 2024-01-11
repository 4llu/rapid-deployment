import torch
import torch.nn as nn
import torch.nn.functional as F


class Explicit_skip_connection(nn.Module):
    def __init__(self, in_channels, out_channels, sigmoid=False):
        super(Explicit_skip_connection, self).__init__()

        self.sigmoid = sigmoid
        self.conv = None
        if in_channels != out_channels:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, x_shortcut):
        z_shortcut = x_shortcut

        if self.conv is not None:
            z_shortcut = self.conv(x_shortcut)

        z = x + z_shortcut
        z = self.bn(z)

        if self.sigmoid:
            return F.sigmoid(z)
        return F.relu(z)


class ModdedInceptionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        reduced_channels,
        use_bottleneck=True,
        use_skip_connection=False,
        use_sen=False,
        activation=True,
        sigmoid=False,
    ):
        super(ModdedInceptionModule, self).__init__()

        self.use_bottleneck = use_bottleneck
        self.use_skip_connection = use_skip_connection
        self.use_sen = use_sen
        self.activation = activation
        self.sigmoid = sigmoid

        self.fc_sen_1 = None
        self.fc_sen_2 = None
        if self.use_sen:
            assert (
                in_channels == reduced_channels * 4
            ), "Input and output channels must match for SEN connection!"
            self.fc_sen_1 = nn.Linear(
                in_channels, in_channels // 4
            )  # * Using reduction factor 4
            self.fc_sen_2 = nn.Linear(in_channels // 4, in_channels)

        self.conv_skip = None
        if self.use_skip_connection and in_channels != reduced_channels * 4:
            self.conv_skip = nn.Conv1d(
                in_channels, reduced_channels * 4, kernel_size=1, stride=1
            )  # , bias=False)

        self.bottleneck = nn.Conv1d(
            in_channels,
            reduced_channels,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.conv_s = nn.Conv1d(
            reduced_channels if self.use_bottleneck else in_channels,
            reduced_channels,
            kernel_size=10,
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv_m = nn.Conv1d(
            reduced_channels if self.use_bottleneck else in_channels,
            reduced_channels,
            kernel_size=20,
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv_l = nn.Conv1d(
            reduced_channels if self.use_bottleneck else in_channels,
            reduced_channels,
            kernel_size=40,
            stride=1,
            padding="same",
            bias=False,
        )

        self.maxpoolpad = nn.ReplicationPad1d((1, 1))
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=0)
        self.conv_maxpool = nn.Conv1d(
            in_channels,
            reduced_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn = nn.BatchNorm1d(reduced_channels * 4)

    def forward(self, x):
        if self.use_bottleneck:
            z_bottleneck = self.bottleneck(x)
        else:
            z_bottleneck = x

        z_maxpool = self.maxpool(x)

        z1 = self.conv_s(z_bottleneck)
        z2 = self.conv_m(z_bottleneck)
        z3 = self.conv_l(z_bottleneck)
        z4 = self.conv_maxpool(self.maxpoolpad(z_maxpool))

        z = torch.concatenate([z1, z2, z3, z4], dim=1)

        if self.use_sen:
            x_channels = F.avg_pool1d(x, x.shape[-1])
            x_channels = x_channels.squeeze()

            x_channels = self.fc_sen_1(x_channels)
            x_channels = F.relu(x_channels)
            x_channels = self.fc_sen_2(x_channels)
            x_channels = F.sigmoid(x_channels)

            x_channels = x_channels.unsqueeze(-1)

            z = z * x_channels

        if self.use_skip_connection:
            # Channel adjust
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            z += x

        if self.activation:
            z = self.bn(z)

            if self.sigmoid:
                z = F.sigmoid(z)
            else:
                z = F.relu(z)
        # else:
        #     z = self.bn(z)
        #     z = F.sigmoid(z)

        return z


class InceptionTime(nn.Module):
    def __init__(self, config):
        super(InceptionTime, self).__init__()

        self.config = config

        self.stem_1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=20, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(16, momentum=1, affine=True),
            nn.ReLU(),
            # nn.MaxPool1d(10, stride=2, padding=0),
        )

        self.module_3_1 = ModdedInceptionModule(16, 4, use_bottleneck=True)
        self.module_4_1 = ModdedInceptionModule(16, 4)
        self.module_5_1 = ModdedInceptionModule(16, 4, activation=False)
        self.shortcut_1 = Explicit_skip_connection(16, 16)

        self.module_6_1 = ModdedInceptionModule(16, 4)
        self.module_7_1 = ModdedInceptionModule(16, 4)
        self.module_8_1 = ModdedInceptionModule(16, 4, activation=False)
        self.shortcut_2 = Explicit_skip_connection(16, 16, sigmoid=False)

    def forward(self, x):
        verbose = False

        if verbose:
            print("Input:", x.shape)

        out = x

        out = self.stem_1(out)
        if verbose:
            print("Stem 1:", out.shape)

        x_earlier = out.clone()

        out = self.module_3_1(out)
        if verbose:
            print("Module 3.1:", out.shape)

        out = self.module_4_1(out)
        if verbose:
            print("Module 4.1:", out.shape)

        out = self.module_5_1(out)
        if verbose:
            print("Module 5.1:", out.shape)

        out = self.shortcut_1(out, x_earlier)
        x_earlier = out.clone()
        if verbose:
            print("Shortcut 1:", out.shape)

        out = self.module_6_1(out)
        if verbose:
            print("Module 6.1:", out.shape)

        out = self.module_7_1(out)
        if verbose:
            print("Module 7.1:", out.shape)

        out = self.module_8_1(out)
        if verbose:
            print("Module 8.1:", out.shape)

        out = self.shortcut_2(out, x_earlier)
        if verbose:
            print("Shortcut 2:", out.shape)

        out = F.avg_pool1d(out, kernel_size=out.shape[-1])
        if verbose:
            print("GAP:", out.shape)

        if verbose:
            quit()

        return out
