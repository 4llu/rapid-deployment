import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLpNormalizer(nn.Module):
    """
    Performs the Lp normalization on the input. Also scales the normalized embedding.
    """

    # Note that torch's own `normalize` only works for positive vectors

    def __init__(self, config):
        super(EmbeddingLpNormalizer, self).__init__()

        self.config = config

        assert self.config["lp_norm"] > 0, "lp_norm must be over 0!"

        def lp_norm(x):
            # ||x||_p = (x_1^p + ... + x_n^p)^(1/p)
            norm = x.abs().pow(self.config["lp_norm"]).sum(dim=-1, keepdim=True) ** (1.0 / self.config["lp_norm"])
            return x / norm

        self.normalizer = lp_norm

    def forward(self, x):
        x = self.normalizer(x)
        return x * self.config["embedding_multiplier"]


class EmbeddingOldNormalizer(nn.Module):
    """
    Legacy
    """

    def __init__(self, config):
        super(EmbeddingOldNormalizer, self).__init__()

        self.config = config

    def forward(self, x):
        return F.normalize(x, p=1.0, dim=1) * self.config["embedding_multiplier"]


class Prototypical(nn.Module):
    def __init__(self, backbone, config):
        super(Prototypical, self).__init__()
        self.config = config
        self.backbone = backbone

        self._prev_query_embeddings = None
        self._prev_prototypes = None

        # Embedding normalizer
        self.embedding_normalizer = None
        if self.config["embedding_normalization_type"] == "lp":
            self.embedding_normalizer = EmbeddingLpNormalizer(config)
        elif self.config["embedding_normalization_type"] == "old":
            self.embedding_normalizer = EmbeddingOldNormalizer(config)
        # elif self.config["embedding_normalization_type"] == "mahalanobis":
        #   self.embedding_normalizer = TODO
        elif self.config["embedding_normalization_type"]:  # If not false
            raise Exception(f"No such embedding normalization as {self.config['embedding_normalize']}")

    def forward(self, support_query):
        # Starts as [n_way, k_shot + n_query, window_length]

        # Add channel dim [n_way, k_shot + n_query, 1, window_length]
        # * Only single sensors inputs supported currently
        # support_query = support_query.unsqueeze(2)
        # Reshape to fit (batch, channel, features) shape
        # [n_way * (k_shot + n_query), 1, window_length]
        support_query = support_query.reshape(-1, *support_query.shape[-2:])

        # Compute embeddings
        embeddings = self.backbone(support_query)

        # Return to original shape (except feature length is now embedding length)
        # [n_way, k_shot + n_query, embedding_len]
        embeddings = embeddings.reshape(
            self.config["n_way"],
            self.config["k_shot"] + self.config["n_query"],
            self.config["embedding_len"],
        )

        # Create prototypes
        support_embeddings = embeddings[:, : self.config["k_shot"]]
        # [n_way, embedding_len]
        prototypes = support_embeddings.mean(dim=1)
        # Normalize prototypes (Important that done after averaging)
        if self.embedding_normalizer:
            prototypes = self.embedding_normalizer(prototypes)

        # Save prototypes
        self._prev_prototypes = torch.clone(prototypes).detach()

        # Query embeddings
        query_embeddings = embeddings[:, self.config["k_shot"] :]
        # Normalize query embeddings
        if self.embedding_normalizer:
            query_embeddings = self.embedding_normalizer(query_embeddings)
        # Save query embeddings
        self._prev_query_embeddings = query_embeddings.clone().detach()
        # Reshape to work with cdist
        query_embeddings = query_embeddings.reshape(-1, self.config["embedding_len"])

        # Calculate distances from each query embedding to each prototype
        # -1 because we wan't shorter distance to be better (bigger prob. after softmax)
        out = -1 * torch.cdist(query_embeddings, prototypes, p=2.0)

        return out, prototypes, query_embeddings
