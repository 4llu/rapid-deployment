import torch


# Methods
#########


def z_score(train_data, validation_data, test_data, config):
    # Statistics only from the training data
    mean = train_data.mean(dim=(0, 2))
    std = train_data.std(dim=(0, 2))

    # (x - mu) / sigma
    train_data = ((train_data.swapaxes(1, 2) - mean) / std).swapaxes(2, 1)
    validation_data = ((validation_data.swapaxes(1, 2) - mean) / std).swapaxes(2, 1)
    test_data = ((test_data.swapaxes(1, 2) - mean) / std).swapaxes(2, 1)

    return train_data, validation_data, test_data


def min_max(train_data, validation_data, test_data, config):
    # * This looks like a weird solution because torch min doesn't
    # * take two dimension inputs (e.g. ´(0, 2)´)

    # Statistics only from the training data
    min = torch.tensor(train_data.numpy().min(axis=(0, 2)))
    max = torch.tensor(train_data.numpy().max(axis=(0, 2)))

    # (x - min) / (max - min)
    train_data = ((train_data.swapaxes(1, 2) - min) / (max - min)).swapaxes(2, 1)
    validation_data = ((validation_data.swapaxes(1, 2) - min) / (max - min)).swapaxes(
        2, 1
    )
    test_data = ((test_data.swapaxes(1, 2) - min) / (max - min)).swapaxes(2, 1)

    return train_data, validation_data, test_data


# Setup
#######


def preprocess_full(train_data, validation_data, test_data, config):
    """
    Preprocess the entire splits as configured. Technically you can do all of these, but most of the
    time only some combinations make sense (or are even possible). Also, the order they are in here matters if some normalizations are
    meant to be run together.

    Parameters:
        train_data : pd.df
        validation_data : pd.df
        test_data : pd.df

    Returns:
        train_data : pd.df
        validation_data : pd.df
        test_data : pd.df
    """

    # Z-score
    if "z-score" in config["preprocessing_full"]:
        train_data, validation_data, test_data = z_score(
            train_data, validation_data, test_data, config
        )

    # Min max
    if "min_max" in config["preprocessing_full"]:
        train_data, validation_data, test_data = min_max(
            train_data, validation_data, test_data, config
        )

    return train_data, validation_data, test_data
