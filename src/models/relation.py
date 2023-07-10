import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.wdcnn import ConvLayer


class L2DistanceNetwork(nn.Module):
    def __init__(self, config):
        super(L2DistanceNetwork, self).__init__()

    def forward(self, x):
        x = torch.sqrt(torch.pow(x[:, :, :256] - x[:, :, 256:], 2))
        x = torch.sum(x, dim=(1, 2))

        return x


class DefaultDistanceNetwork(nn.Module):
    def __init__(self, config):
        super(DefaultDistanceNetwork, self).__init__()
        self.config = config

        self.cn_layer1 = ConvLayer(
            64, 64,
            kernel_size=3,
            stride=1,
            padding="same",
            dropout=config["cl_dropout"],
        )
        self.cn_layer2 = ConvLayer(
            64, 8,
            kernel_size=3,
            stride=1,
            padding="same",
            dropout=config["cl_dropout"],
        )

        self.fc1 = nn.Linear(128*8, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        verbose = False

        if verbose:
            print("Input:", x.shape)

        # Conv layers

        out = self.cn_layer1(x)
        if verbose:
            print("CL 1:", out.shape)

        out = self.cn_layer2(out)
        if verbose:
            print("CL 2:", out.shape)

        # Flatten channels
        out = out.view(out.shape[0], -1)
        if verbose:
            print(out.shape)

        out = self.fc1(out)
        out = F.relu(out)
        if verbose:
            print(out.shape)

        out = self.fc2(out)
        out = F.sigmoid(out)
        if verbose:
            print(out.shape)

        return out


class Relation(nn.Module):
    def __init__(self, backbone, config):
        super(Relation, self).__init__()
        self.config = config
        self.backbone = backbone

        self.distance_network = L2DistanceNetwork(config)
        # self.distance_network = None
        # if config["distance_network"] == "default":
        #     self.distance_network = DefaultDistanceNetwork(config)

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
        #                     [prototypes[k, :, :], query_embeddings[j, i, :, :]], dim=-1)
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

        support_query = torch.cat([prototypes, query_embeddings], dim=-1)
        # [n_way * n_query * n_way, embedding_channels, embedding_len * 2]

        distances = self.distance_network(support_query)

        # print(distances.shape)
        # distances = distances.flatten()
        distances = distances.reshape(self.config["n_way"],
                                      self.config["n_query"],
                                      self.config["n_way"]
                                      )
        return distances
