import torch
import torch.nn as nn

from models.distance_networks.relation_default_distance import DefaultDistanceNetwork
from models.distance_networks.simple_distance import SimpleDistanceNetwork

class L2DistanceNetwork(nn.Module):
    def __init__(self, config):
        super(L2DistanceNetwork, self).__init__()

    def forward(self, x):
        x = torch.abs(x[:, 0, :] - x[:, 1, :])
        # x = torch.abs(x[:, :, :128] - x[:, :, 128:])
        # x = torch.sqrt(torch.pow(x[:, :, :128] - x[:, :, 128:], 2))
        x = torch.sum(x, dim=-1)
        # x = torch.sum(x, dim=(1, 2))
        x = -x

        return x

class Relation(nn.Module):
    def __init__(self, backbone, distance_network, config):
        super(Relation, self).__init__()
        self.config = config
        self.backbone = backbone
        self.distance_network = distance_network

        # self.distance_network = L2DistanceNetwork(config)

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
        embeddings = embeddings.unsqueeze(1) # XXX Remove (only for wdcnn)

        # Return to original shape (except feature length is now embedding length)
        # [n_way, k_shot + n_query, embedding_len]
        embeddings = embeddings.reshape(
            self.config["n_way"],
            self.config["k_shot"] + self.config["n_query"],
            embeddings.shape[-2],  # Embedding channels
            embeddings.shape[-1],  # Embedding feature length
        )

        # Create prototypes
        support_embeddings = embeddings[:, : self.config["k_shot"]]
        # [n_way, k_shot, embedding_channels, embedding_len]
        prototypes = support_embeddings.mean(dim=1)
        # [n_way, embedding_channels, embedding_len]
        prototypes = prototypes.repeat(400, 1, 1)

        # Query embeddings
        query_embeddings = embeddings[:, self.config["k_shot"]:]
        # [n_way, n_query, embedding_channels, embedding_len]

        # print("-----")
        # print(prototypes[0, 0, :10])
        # print("-----")
        # for i in range(4):
        #     print(query_embeddings[0, i, 0, :10])
        #     print()

        # print(">>>>>>")

        # print("-----")
        # print(prototypes[9, 0, :10])
        # print("-----")
        # for i in range(4):
        #     print(query_embeddings[9, i, 0, :10])
        #     print()
        # quit()

        ######

        # c = []
        # for j in range(10):
        #     for i in range(40):
        #         for k in range(10):
        #             c.append(
        #                 torch.cat(
        #                     [prototypes[k, :, :], query_embeddings[j, i, :, :]], dim=-2)
        #                     # [prototypes[k, :, :], query_embeddings[j, i, :, :]], dim=-1)
        #             )

        # c = torch.stack(c)
        # distances = self.distance_network(c)

        # distances = distances.reshape(self.config["n_way"],
        #                               self.config["n_query"],
        #                               self.config["n_way"]
        #                               )

        # return distances

        ######

        query_embeddings = query_embeddings.reshape(-1,
                                                    *query_embeddings.shape[-2:])
        query_embeddings = query_embeddings.repeat_interleave(10, dim=0)
        # print(query_embeddings.shape)

        support_query = torch.cat([prototypes, query_embeddings], dim=-2) # Depth-wise
        # support_query = torch.cat([prototypes, query_embeddings], dim=-1) # Length-wise
        # [n_way * n_query * n_way, embedding_channels, embedding_len * 2]

        distances = self.distance_network(support_query)

        # print(distances.shape)
        # distances = distances.flatten()
        distances = distances.reshape(self.config["n_way"],
                                      self.config["n_query"],
                                      self.config["n_way"]
                                      )
        
        # print("----------------")
        # print()
        # for i in range(4):
        #     print(distances[0, i, :])
        #     print()

        # print(">>>>>>")

        # for i in range(4):
        #     print(distances[0, i, :])
        #     print()

        # quit()

        return distances
