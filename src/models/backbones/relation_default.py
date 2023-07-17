import torch.nn as nn


class RelationDefault(nn.Module):
    def __init__(self, config):
        super(RelationDefault, self).__init__()
        self.config = config
        self.kernel_size = 9

        # Convolutional layers
        # self.cn_layer0 = nn.Sequential(
        #     nn.Conv1d(1, 16, kernel_size=7, padding="same", bias=False),
        #     nn.BatchNorm1d(16, momentum=1, affine=True),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2),
        # )
        self.cn_layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=self.kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(16, momentum=1, affine=True),
            # nn.ReLU(),
            nn.Hardswish(),
            # nn.MaxPool1d(2),
        )
        self.cn_layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=self.kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(32, momentum=1, affine=True),
            # nn.ReLU(),
            nn.Hardswish(),
            # nn.MaxPool1d(2),
        )
        self.cn_layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=self.kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            # nn.ReLU(),
            nn.Hardswish(),
            # nn.MaxPool1d(2),
        )
        self.cn_layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=self.kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            # nn.ReLU(),
            nn.Hardswish(),
            # nn.MaxPool1d(2),
        )

    def forward(self, x):
        verbose = False

        if verbose:
            print("Input:", x.shape)

        out = x

        # Conv layers

        # out = self.cn_layer0(out)
        # if verbose:
        #     print("CL 0:", out.shape)

        out = self.cn_layer1(out)
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
