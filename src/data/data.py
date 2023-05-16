from data.arotor import get_arotor_data
from data.arotor_old import get_arotor_old_data


# SETUP
#######


def setup_data(config, device):
    print("PREPARING DATA")

    # DATA INITIALIZATION
    #####################
    if config["data"] == "ARotor":
        train_loader, validation_loader, test_loader = get_arotor_data(config, device)
    elif config["data"] == "ARotor_old":
        train_loader, validation_loader, test_loader = get_arotor_old_data(config, device)
    else:
        raise Exception("No such data configuration as`", config["data"], "`!")

    print("PREPARING DATA DONE")

    return train_loader, validation_loader, test_loader
