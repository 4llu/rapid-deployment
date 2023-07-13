import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)  # Padding depends on stride size
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


class ModdedInceptionModule(nn.Module):
    def __init__(self, in_channels, reduced_channels, shortcut_connection=False):  # list
        super(ModdedInceptionModule, self).__init__()

        self.shortcut_connection = shortcut_connection

        self.bottleneck = nn.Conv1d(
            in_channels,
            reduced_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv_s = nn.Conv1d(
            in_channels,
            reduced_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv_m = nn.Conv1d(
            reduced_channels,
            reduced_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv_l = nn.Conv1d(
            reduced_channels,
            reduced_channels,
            kernel_size=12,
            stride=1,
            padding="same",
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)  # Padding depends on stride size
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
        z_bottleneck = self.bottleneck(x)
        z_maxpool = self.maxpool(x)

        z1 = self.conv_s(x)
        z2 = self.conv_m(z_bottleneck)
        z3 = self.conv_l(z_bottleneck)
        z4 = self.conv_maxpool(z_maxpool)

        z = torch.concatenate([z1, z2, z3, z4], dim=1)

        if self.shortcut_connection:
            z += x

        z = self.bn(z)
        # z = F.relu(z)
        z = F.hardswish(z)  # TODO Try

        return z


class GridReductionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        reduced_channels,
    ):
        super(GridReductionModule, self).__init__()

        self.bottleneck = nn.Conv1d(
            in_channels,
            reduced_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv_1 = nn.Conv1d(
            reduced_channels,
            reduced_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.conv_2 = nn.Conv1d(
            reduced_channels,
            reduced_channels,
            kernel_size=9,
            stride=2,
            padding=4,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(2, stride=2, padding=0)  # Padding depends on stride size

        self.bn = nn.BatchNorm1d(reduced_channels * 2 + in_channels)

    def forward(self, x):
        z_bottleneck = self.bottleneck(x)
        z_maxpool = self.maxpool(x)

        z1 = self.conv_1(z_bottleneck)
        z2 = self.conv_2(z_bottleneck)

        z = torch.concatenate([z1, z2, z_maxpool], dim=1)

        z = self.bn(z)
        # z = F.relu(z)
        z = F.hardswish(z)  # TODO Try

        return z


class SimpleGridReductionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_out_channels,
    ):
        super(SimpleGridReductionModule, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            conv_out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)

        self.bn = nn.BatchNorm1d(conv_out_channels + in_channels)

    def forward(self, x):
        z1 = self.conv(x)
        z_maxpool = self.maxpool(x)

        z = torch.concatenate([z1, z_maxpool], dim=1)

        z = self.bn(z)
        # z = F.relu(z)
        z = F.hardswish(z)  # TODO Try

        return z


class InceptionTime(nn.Module):
    def __init__(self, config):
        super(InceptionTime, self).__init__()

        self.config = config

        # Layers
        self.stem_1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=16, stride=4, padding=0, bias=False),
            nn.BatchNorm1d(16, momentum=1, affine=True),
            # nn.ReLU(),
            nn.Hardswish(),
        )

        self.stem_reduction_1 = SimpleGridReductionModule(16, 16)
        # self.module_1_1 = ModdedInceptionModule(32, 8, shortcut_connection=True)

        self.stem_reduction_2 = SimpleGridReductionModule(32, 32)
        # self.module_2_1 = ModdedInceptionModule(64, 16, shortcut_connection=True)

        self.stem_reduction_3 = SimpleGridReductionModule(64, 64)
        # self.module_3_1 = ModdedInceptionModule(128, 32, shortcut_connection=True)
        # self.stem_reduction_2 = SimpleGridReductionModule(32, 32)

        # self.module_1_2 = ModdedInceptionModule(32, 8)
        # self.module_1_reduction = SimpleGridReductionModule(32, 32)
        # # self.module_1_reduction = GridReductionModule(32, 16)

        # self.module_2_2 = ModdedInceptionModule(64, 16)
        # self.module_2_reduction = SimpleGridReductionModule(64, 64)
        # # self.module_2_reduction = GridReductionModule(64, 32)

        # self.module_3_2 = ModdedInceptionModule(128, 32)
        # # self.module_3_reduction = SimpleGridReductionModule(128, 64)
        # # self.module_3_reduction = GridReductionModule(128, 64)

    def forward(self, x):
        verbose = False

        if verbose:
            print("Input:", x.shape)

        out = x

        out = self.stem_1(out)
        if verbose:
            print("C1:", out.shape)

        out = self.stem_reduction_1(out)
        if verbose:
            print("Reduction 1:", out.shape)

        # out = self.module_1_1(out)
        # if verbose:
        #     print("Module 1.1:", out.shape)

        out = self.stem_reduction_2(out)
        if verbose:
            print("Reduction 2:", out.shape)

        # out = self.module_2_1(out)
        # if verbose:
        #     print("Module 2.1:", out.shape)

        out = self.stem_reduction_3(out)
        if verbose:
            print("Reduction 3:", out.shape)

        # out = self.module_3_1(out)
        # if verbose:
        #     print("Module 3.1:", out.shape)

        return out


class InceptionTime_tmp(nn.Module):
    def __init__(self, config):
        super(InceptionTime, self).__init__()

        self.config = config

        # Layers
        self.stem_1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16, momentum=1, affine=True),
            # nn.ReLU(),
            nn.Hardswish(),
        )

        self.stem_reduction_1 = SimpleGridReductionModule(16, 16)
        # self.stem_reduction_2 = SimpleGridReductionModule(32, 32)

        self.module_1_1 = ModdedInceptionModule(32, 8)
        self.module_1_2 = ModdedInceptionModule(32, 8)
        self.module_1_reduction = SimpleGridReductionModule(32, 32)
        # self.module_1_reduction = GridReductionModule(32, 16)

        self.module_2_1 = ModdedInceptionModule(64, 16)
        self.module_2_2 = ModdedInceptionModule(64, 16)
        # self.module_2_reduction = SimpleGridReductionModule(64, 64)
        # self.module_2_reduction = GridReductionModule(64, 32)

        # self.module_3_1 = ModdedInceptionModule(128, 32)
        # self.module_3_2 = ModdedInceptionModule(128, 32)
        # self.module_3_reduction = SimpleGridReductionModule(128, 64)
        # self.module_3_reduction = GridReductionModule(128, 64)

    def forward(self, x):
        verbose = False

        if verbose:
            print("Input:", x.shape)

        out = x

        out = self.stem_1(out)
        if verbose:
            print("C1:", out.shape)

        out = self.stem_reduction_1(out)
        if verbose:
            print("Reduction 1:", out.shape)
        # out = self.stem_reduction_2(out)
        # if verbose:
        #     print("Reduction 2:", out.shape)

        # 1

        out = self.module_1_1(out)
        if verbose:
            print("1.1:", out.shape)

        out = self.module_1_2(out)
        if verbose:
            print("1.2:", out.shape)

        out = self.module_1_reduction(out)
        if verbose:
            print("1.reduction:", out.shape)

        # 2

        out = self.module_2_1(out)
        if verbose:
            print("2.1:", out.shape)

        out = self.module_2_2(out)
        if verbose:
            print("2.2:", out.shape)

        out = self.module_2_reduction(out)
        if verbose:
            print("2.reduction:", out.shape)

        # 3

        out = self.module_3_1(out)
        if verbose:
            print("3.1:", out.shape)

        out = self.module_3_2(out)
        if verbose:
            print("3.2:", out.shape)

        # out = self.module_3_reduction(out)
        # if verbose:
        #     print("3.reduction:", out.shape)

        return out


class InceptionTime_old(nn.Module):
    def __init__(self, config):
        super(InceptionTime_old, self).__init__()

        self.config = config

        # Convolutional layers
        self.module_1 = InceptionModule(1, 32)
        self.module_2 = InceptionModule(32 * 4, 64)
        # self.module_3 = InceptionModule(32 * 4, 64)

        # Global average pooling
        self.globalAvgPool = nn.AvgPool1d(kernel_size=722)  # FIXME Kernel size (Depends on input length)

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
