import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config

        # Classifier
        self.lstm = nn.LSTM(
            1,
            64,
            3,
            batch_first=True,
            # proj_size=config["embedding_len"],
        )

        # Classifier
        self.fc1 = nn.Linear(
            64,
            self.config["embedding_len"],
        )

    def forward(self, x):
        verbose = False

        if verbose:
            print("INPUT:", x.shape)

        x = torch.transpose(x, 1, 2)
        if verbose:
            print("INPUT swapped:", x.shape)

        out, _ = self.lstm(x)
        if verbose:
            print("LSTM out:", out.shape)

        out = out[:, -1, :]
        if verbose:
            print("LSTM out last:", out.shape)

        out = self.fc1(out)
        if verbose:
            print("FC1:", out.shape)

        if verbose:
            quit()

        return out
