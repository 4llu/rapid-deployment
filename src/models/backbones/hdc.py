import torch
import torch.nn as nn
import torch.nn.functional as F


class HDCModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilation_pattern=[1, 2, 3],
        use_bn=True,
        use_res=False,
        use_sen=False,
        last=False,
    ):  # list
        super(HDCModule, self).__init__()

        self.dilation_pattern = dilation_pattern
        self.use_bn = use_bn
        self.use_res = use_res
        self.use_sen = use_sen
        self.last = last

        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=self.dilation_pattern[0],
            padding="same",
            bias=not self.use_bn,
        )
        if self.use_bn:
            self.bn_1 = nn.BatchNorm1d(out_channels)

        self.conv_2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=self.dilation_pattern[1],
            padding="same",
            bias=not self.use_bn,
        )
        if self.use_bn:
            self.bn_2 = nn.BatchNorm1d(out_channels)

        self.conv_3 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=self.dilation_pattern[2],
            padding="same",
            bias=not self.use_bn,
        )

        if self.use_bn:  # and not self.last:
            self.bn_3 = nn.BatchNorm1d(out_channels)

        # SEN
        if self.use_sen:
            assert in_channels == out_channels, "Input and output channels must match for SEN connection!"
            self.fc_sen_1 = nn.Linear(in_channels, in_channels // 4)  # * Using reduction factor 4
            self.fc_sen_2 = nn.Linear(in_channels // 4, in_channels)

        # RES
        self.conv_res = None
        if self.use_res and in_channels != out_channels:
            self.conv_res = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding="same",
                bias=not self.use_bn,
            )

    def forward(self, x):
        verbose = False

        if verbose:
            print("Input:", x.shape)

        out = x

        # Conv layers

        # 1
        out = self.conv_1(out)
        if self.use_bn:
            out = self.bn_1(out)
        out = F.relu(out)
        if verbose:
            print("Conv 1:", out.shape)

        # 2
        out = self.conv_2(out)
        if self.use_bn:
            out = self.bn_2(out)
        out = F.relu(out)
        if verbose:
            print("Conv 2:", out.shape)

        # 3
        out = self.conv_3(out)
        if verbose:
            print("Conv 3:", out.shape)

        # SEN
        if self.use_sen:
            out_sen = F.avg_pool1d(x, x.shape[-1])
            if verbose:
                print("     SEN GAP", out_sen.shape)

            out_sen = self.fc_sen_1(out_sen)
            out_sen = F.relu(out_sen)
            if verbose:
                print("     SEN FC1", out_sen.shape)

            out_sen = self.fc_sen_2(out_sen)
            out_sen = F.sigmoid(out_sen)
            if verbose:
                print("     SEN FC2", out_sen.shape)

            # Channel wise multiplication
            out *= out_sen

        # RES
        if self.use_res:
            if self.conv_res is None:
                out += x
            else:
                out += self.conv_res(x)

        # Conv 3 BN + activation
        if not self.last:
            if self.use_bn:
                out = self.bn_3(out)
            out = F.relu(out)
        else:
            if self.use_bn:
                out = self.bn_3(out)
            out = F.leaky_relu(out) * 10

        return out


class HDC(nn.Module):
    def __init__(self, config):
        super(HDC, self).__init__()
        self.config = config

        # TODO Convert to config option
        use_bn = True
        use_max_pool = False
        dilation_pattern = [1, 3, 5]
        # dilation_pattern = [1, 2, 3]

        # Stem

        # stem = []
        # stem.append(
        #     nn.Conv1d(
        #         1,
        #         16,
        #         kernel_size=64,
        #         stride=2,
        #         bias=False,
        #     )
        # )
        # stem.append(nn.ReLU())
        # if use_bn:
        #     stem.append(nn.BatchNorm1d(16))
        # self.stem = nn.Sequential(*stem)

        # HDC modules

        module_16_1 = HDCModule(1, 16, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False)
        module_16_2 = HDCModule(16, 16, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False)
        module_16_3 = HDCModule(16, 16, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False)

        if use_max_pool:
            maxpool_1 = nn.MaxPool1d(2, 2)

        module_32_1 = HDCModule(16, 32, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False)
        module_32_2 = HDCModule(32, 32, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False)
        module_32_3 = HDCModule(32, 32, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False)
        # module_32_4 = HDCModule(
        #     32, 32, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False
        # )

        if use_max_pool:
            maxpool_2 = nn.MaxPool1d(2, 2)

        module_64_1 = HDCModule(32, 64, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False)
        module_64_2 = HDCModule(64, 64, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False)
        module_64_3 = HDCModule(
            64, 64, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False, last=True
        )
        # module_64_4 = HDCModule(
        #     64, 64, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False
        # )

        # if use_max_pool:
        #     maxpool_3 = nn.MaxPool1d(2, 2)

        # module_128_1 = HDCModule(64, 128, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False)
        # module_128_2 = HDCModule(
        #     128, 128, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False
        # )
        # module_128_3 = HDCModule(
        #     128, 128, dilation_pattern=dilation_pattern, use_bn=use_bn, use_res=True, use_sen=False, last=False
        # )

        self.blocks = []

        block_16 = nn.Sequential(
            module_16_1,
            module_16_2,
            module_16_3,
        )
        self.blocks.append(block_16)

        if use_max_pool:
            self.blocks.append(maxpool_1)

        block_32 = nn.Sequential(
            module_32_1,
            module_32_2,
            module_32_3,
        )
        self.blocks.append(block_32)

        if use_max_pool:
            self.blocks.append(maxpool_2)

        block_64 = nn.Sequential(
            module_64_1,
            module_64_2,
            module_64_3,
        )
        self.blocks.append(block_64)

        # if use_max_pool:
        #     self.blocks.append(maxpool_3)

        # block_128 = nn.Sequential(
        #     module_128_1,
        #     module_128_2,
        #     module_128_3,
        # )
        # self.blocks.append(block_128)

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        verbose = False

        if verbose:
            print("Input:", x.shape)

        out = x

        # Layers
        # out = self.stem(out)
        # if verbose:
        #     print("AFTER STEM:", out.shape)

        out = self.blocks(out)
        if verbose:
            print("AFTER BLOCKS:", out.shape)

        out = F.avg_pool1d(out, out.shape[-1])
        out = out.squeeze()

        if verbose:
            print("OUT:", out.shape)
            quit()

        return out
