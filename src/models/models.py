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
    else:
        raise Exception(f"No such backbone name as: {config['backbone']}!")

    # Model selection
    #################

    model = None
    if config["model"] == "prototypical":
        from models.prototypical import Prototypical

        model = Prototypical(backbone, config)
    else:
        raise Exception(f"No such model name as: {config['model']}!")

    # Move model to correct device
    model = model.to(device)

    # * torch.compile doesn't work on windows currently
    if device.type == "cuda" and platform.system() != "Windows":
        model = torch.compile(model)

    return model
