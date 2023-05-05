import torch
import torch.nn as nn


class Prototypical(nn.Module):
    def __init__(self, backbone, config):
        super(Prototypical, self).__init__()
        self.config = config
        self.backbone = backbone

        self._prev_query_embeddings = None
        self._prev_prototypes = None

    def forward(self, support_query):
        # Starts as [n_way, k_shot + n_query, window_length]

        # Add channel dim [n_way, k_shot + n_query, 1, window_length]
        # * Only single sensors inputs supported currently
        support_query = support_query.unsqueeze(2)
        # Reshape to fit (batch, channel, features) shape
        # [n_way * (k_shot + n_query), 1, window_length]
        support_query = support_query.reshape(-1, 1, support_query.size(-1))

        # Compute embeddings
        embeddings = self.backbone(support_query)
        print("Some sum", embeddings[0].shape)

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
        # Save prototypes
        self._prev_prototypes = torch.clone(prototypes).detach()

        # Query embeddings
        query_embeddings = embeddings[:, self.config["k_shot"] :]
        # Save query embeddings
        self._prev_query_embeddings = query_embeddings.clone().detach()
        # Reshape to work with cdist
        query_embeddings = query_embeddings.reshape(-1, self.config["embedding_len"])

        # Calculate distances from each query embedding to each prototype
        # -1 because we wan't shorter distance to be better (bigger prob. after softmax)
        print("Q", query_embeddings.shape)
        print("Q sum", query_embeddings[0].sum())
        print("P", prototypes.shape)
        print("P sum", prototypes[0].sum())

        out = -1 * torch.cdist(query_embeddings, prototypes, p=2.0)

        return out  # , prototypes
