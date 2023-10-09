import platform

import torch


def setup_model(config, device):

    # Backbone selection
    ####################

    backbone = None
    if config["backbone"] == "WDCNN":
        from models.backbones.wdcnn import WDCNN

        backbone = WDCNN(config)
    elif config["backbone"] == "WDCNN_old":
        from models.backbones.wdcnn_old import WDCNN_old

        backbone = WDCNN_old(config)
    elif config["backbone"] == "MiCNN":
        from models.backbones.micnn import MiCNN

        backbone = MiCNN(config)
    elif config["backbone"] == "HDC":
        from models.backbones.hdc import HDC

        backbone = HDC(config)
    elif config["backbone"] == "mininet":
        from models.backbones.mininet import mininet

        backbone = mininet(config)
    elif config["backbone"] == "InceptionTime":
        from models.backbones.inception_time import InceptionTime

        backbone = InceptionTime(config)
    elif config["backbone"] == "RelationDefault":
        from models.backbones.relation_default import RelationDefault

        backbone = RelationDefault(config)
    else:
        raise Exception(f"No such backbone name as: {config['backbone']}!")

    # Model selection
    #################

    model = None
    if config["model"] == "prototypical":
        from models.prototypical import Prototypical

        model = Prototypical(backbone, config)
    elif config["model"] == "relation":
        from models.relation import Relation

        # Distance network selection
        #################

        distance_network = None
        if config["distance_network"] == "default":
            from models.distance_networks.relation_default_distance import DefaultDistanceNetwork

            distance_network = DefaultDistanceNetwork(config)
        elif config["distance_network"] == "simple":
            from models.distance_networks.simple_distance import SimpleDistanceNetwork

            distance_network = SimpleDistanceNetwork(config)
        elif config["distance_network"] == "very_simple":
            from models.distance_networks.very_simple_distance import VerySimpleDistanceNetwork

            distance_network = VerySimpleDistanceNetwork(config)
        elif config["distance_network"] == "HDCD":
            from models.distance_networks.hdc_distance import HDCDistanceNetwork

            distance_network = HDCDistanceNetwork(config)

        model = Relation(backbone, distance_network, config)

    else:
        raise Exception(f"No such model name as: {config['model']}!")

    # Print here so model details can be seen in logs
    # print(model)
    # Move model to correct device
    model = model.to(device)

    # * torch.compile doesn't work on windows currently
    if device.type == "cuda" and platform.system() != "Windows":
        model = torch.compile(model)

    # Print model
    # rows = []
    # t_params = 0
    # for name, parameter in model.named_parameters():
    #     if not parameter.requires_grad:
    #         continue

    #     param = parameter.numel()
    #     rows.append([name, param])
    #     t_params += param

    # for r in rows:
    #     print("{:<35} {:<10}".format(r[0], r[1]))
    # print("Total parameters:", t_params)
    # quit()

    return model
