import torch
import torch.nn as nn

from models.distance_networks.relation_default_distance import DefaultDistanceNetwork
from models.distance_networks.simple_distance import SimpleDistanceNetwork


# class L2DistanceNetwork(nn.Module):
#     def __init__(self, config):
#         super(L2DistanceNetwork, self).__init__()

#     def forward(self, x):
#         x = torch.abs(x[:, 0, :] - x[:, 1, :])
#         x = torch.sum(x, dim=-1)
#         x = -x

#         return x


class Relation(nn.Module):
    def __init__(self, backbone, distance_network, config):
        super(Relation, self).__init__()
        self.config = config
        self.backbone = backbone
        self.distance_network = distance_network

        # self.distance_network = L2DistanceNetwork(config)

    def forward(self, support_query):
        # support_query: [n_way, k_shot + n_query, window_length]

        # Check dimensions
        # if len(support_query.shape) == 3:
        #     support_query = support_query.unsqueeze(2)

        assert len(support_query.shape) == 4, "Support_query set has the wrong number of dimensions: {}".format(
            len(support_query.shape)
        )

        # Create embeddings
        ##

        support_query = support_query.reshape(support_query.shape[0] * support_query.shape[1], *support_query.shape[2:])
        support_query_embeddings = self.backbone(support_query)

        # FIXME Ensure all backbones have channel support
        if len(support_query_embeddings.shape) == 2:
            support_query_embeddings = support_query_embeddings.unsqueeze(1)

        # Return to original shape
        support_query_embeddings = support_query_embeddings.reshape(
            self.config["n_way"], self.config["k_shot"] + self.config["n_query"], *support_query_embeddings.shape[1:]
        )

        # Support embeddings summing
        support_embeddings = support_query_embeddings[:, : self.config["k_shot"], :, :]
        # Combine k_shot support samples
        support_embeddings = support_embeddings.sum(dim=1)  # XXX How about mean?

        # Query embedding reshaping
        query_embeddings = support_query_embeddings[:, self.config["k_shot"] :, :, :]
        query_embeddings = query_embeddings.reshape(
            query_embeddings.shape[0] * query_embeddings.shape[1], *query_embeddings.shape[2:]
        )

        # Calculcate relations
        ##

        support_embeddings_ext = support_embeddings.unsqueeze(0).repeat(
            self.config["n_way"] * self.config["n_query"], 1, 1, 1
        )

        query_embeddings_ext = query_embeddings.unsqueeze(0).repeat(self.config["n_way"], 1, 1, 1)
        query_embeddings_ext = torch.transpose(query_embeddings_ext, 0, 1)

        # Create pairs
        relation_pairs = torch.cat([support_embeddings_ext, query_embeddings_ext], dim=2)
        relation_pairs = relation_pairs.reshape(
            relation_pairs.shape[0] * relation_pairs.shape[1], *relation_pairs.shape[2:]
        )

        # Relation scores
        distances = self.distance_network(relation_pairs).reshape(-1, self.config["n_way"])

        return distances
