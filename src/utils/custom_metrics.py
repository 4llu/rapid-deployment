import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


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

        y_pred = torch.argmax(y_pred, dim=-1)  # .flatten()
        y = torch.argmax(y, dim=-1)  # .flatten

        self.TPTN += (y_pred == y).sum()
        self.N += torch.numel(y_pred)

    def compute(self):
        if self.N == 0:
            raise NotComputableError(
                "FSAccuracy must have at least one example before it can be computed."
            )

        return self.TPTN.item() / self.N
