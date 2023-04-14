import platform

import torch


def setup_model(config, device):
    model = None

    if config["model"] == "wdcnn":
        from models.wdcnn import WDCNN

        model = WDCNN(len(config["signals"]), 61, config)

    elif config["model"] == "micnn":
        from models.micnn import MiCNN

        model = MiCNN(len(config["signals"]), 61, config)

    else:
        raise Exception(f"No such model name as: {config['model']}!")

    model = model.to(device)
    if device.type == "cuda" and platform.system() != "Windows":
        model = torch.compile(model)
    return model
