import torch


class WhiteNoise:
    def __init__(self, std, spacing):
        self.std = std
        self.spacing = spacing  # Spacing = 0 means no dilation

    def __call__(self, batch):
        noise_base = torch.randn([*batch.size()[:-1], int(batch.size(-1) / (self.spacing + 1))])
        # Scale size if spaced
        noise = torch.nn.functional.interpolate(input=noise_base, size=batch.size(-1), mode="linear")
        # Scale
        batch = batch + noise * self.std

        return batch

    def __repr__(self):
        return self.__class__.__name__ + "(std={0}, spacing={1})".format(self.std, self.spacing)


class AugmentationCollation:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, batch):
        samples = torch.stack([s[0] for s in batch])
        labels = torch.tensor([s[1] for s in batch])

        if self.transforms:
            samples = self.transforms(samples)

        return samples, labels
