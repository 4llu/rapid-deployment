import numpy as np
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from sklearn.metrics import confusion_matrix


class RNFSAccuracy(Metric):
    def __init__(self, device):
        # self.n_way = n_way
        self.device = device

        self.TPTN = torch.tensor(0, device=self.device)
        self.N = 0

        super(RNFSAccuracy, self).__init__()

    def reset(self):
        self.TPTN = torch.tensor(0, device=self.device)
        self.N = 0

        super(RNFSAccuracy, self).reset()

    @torch.no_grad()  # ? Possibly unnecessary, but shouldn't be harmful either
    def update(self, output):
        y_pred, y = output[0], output[1]

        y_pred = torch.argmax(y_pred, dim=-1)
        y = torch.argmax(y, dim=-1)

        self.TPTN += (y_pred == y).sum()
        self.N += torch.numel(y_pred)

    def compute(self):
        if self.N == 0:
            raise NotComputableError(
                "FSAccuracy must have at least one example before it can be computed."
            )

        return self.TPTN.item() / self.N


class Confusion_matrices(Metric):
    def __init__(self):
        self.y_pred = []
        self.y = []

        super(Confusion_matrices, self).__init__()

    def reset(self):
        self.y_pred = []
        self.y = []

        super(Confusion_matrices, self).reset()

    @torch.no_grad()  # ? Possibly unnecessary, but shouldn't be harmful either
    def update(self, output):
        y_pred = torch.argmax(output[0].cpu(), dim=-1)
        y = output[1].cpu()

        self.y_pred.extend(y_pred)
        self.y.extend(y)

    def compute(self):
        if len(self.y) == 0:
            raise NotComputableError(
                "Confusion_matrices must have at least one example before it can be computed."
            )

        return self.y, self.y_pred

        # cf = confusion_matrix(self.y, self.y_pred)
        # return confusion_matrix(self.y, self.y_pred)
