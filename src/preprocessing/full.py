import torch


# Methods
#########


# def z_score(train_data, validation_data, test_data, config):
#     # Statistics only from the training data
#     mean = train_data.mean(dim=(0, 2))
#     std = train_data.std(dim=(0, 2))

#     # (x - mu) / sigma
#     train_data = ((train_data.swapaxes(1, 2) - mean) / std).swapaxes(2, 1)
#     validation_data = ((validation_data.swapaxes(1, 2) - mean) / std).swapaxes(2, 1)
#     test_data = ((test_data.swapaxes(1, 2) - mean) / std).swapaxes(2, 1)

#     return train_data, validation_data, test_data


# def min_max(train_data, validation_data, test_data, config):
#     # * This looks like a weird solution because torch min doesn't
#     # * take two dimension inputs (e.g. ´(0, 2)´)

#     # Statistics only from the training data
#     min = torch.tensor(train_data.numpy().min(axis=(0, 2)))
#     max = torch.tensor(train_data.numpy().max(axis=(0, 2)))

#     # (x - min) / (max - min)
#     train_data = ((train_data.swapaxes(1, 2) - min) / (max - min)).swapaxes(2, 1)
#     validation_data = ((validation_data.swapaxes(1, 2) - min) / (max - min)).swapaxes(
#         2, 1
#     )
#     test_data = ((test_data.swapaxes(1, 2) - min) / (max - min)).swapaxes(2, 1)

#     return train_data, validation_data, test_data


def mixed_query_normalization(train_data, validation_data, test_data, config):
    head_len = 3012 * 10

    train_grouped = train_data.groupby(["class", "rpm"])

    # Separate head (to be used for scaling, etc.) from measurements to be used for training/validation/testing (tail)
    # * Take the same amount away from fault measurements too, to keep class balance
    train_head = train_grouped.head(head_len)
    train_tail = train_grouped.tail(
        int(
            (
                len(train_data)
                / (len(config["train_classes"]) * len(config["train_rpms"]))
            )
            - head_len
        )
    )

    # SCALING #
    # Robust scaling with 25 and 75 percentiles

    # Get scales
    scale = {}
    train_head_grouped = train_head.groupby(["class", "rpm"])
    for n, g in train_head_grouped:
        if n[0] == 0:
            p25 = g[config["train_sensors"]].quantile(0.25)
            p75 = g[config["train_sensors"]].quantile(0.75)
            scale[n[1]] = p75 - p25

    # Scale head
    for n, g in train_head_grouped:
        g[config["train_sensors"]] = g[config["train_sensors"]] / scale[n[1]]

    # Scale tail
    for n, g in train_tail.groupby(["class", "rpm"]):
        g[config["train_sensors"]] = g[config["train_sensors"]] / scale[n[1]]

    # MASK CALCULATION #


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
    if "mixed_query_normalization" in config["preprocessing_full"]:
        train_data, validation_data, test_data = mixed_query_normalization(
            train_data, validation_data, test_data, config
        )

    # # Z-score
    # if "z-score" in config["preprocessing_full"]:
    #     train_data, validation_data, test_data = z_score(
    #         train_data, validation_data, test_data, config
    #     )

    # # Min max
    # if "min_max" in config["preprocessing_full"]:
    #     train_data, validation_data, test_data = min_max(
    #         train_data, validation_data, test_data, config
    #     )

    return train_data, validation_data, test_data
