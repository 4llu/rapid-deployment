import numpy as np
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
    masks = {}

    def mixed_query_normalization_helper(data, split):
        classes = config[f"{split}_classes"]
        sensors = config[f"{split}_sensors"]
        rpms = config[f"{split}_rpm"]

        data_grouped = data.groupby(["class", "rpm"])

        # Separate head (to be used for scaling, etc.) from measurements to be used for training/validation/testing (tail)
        # * Take the same amount away from fault measurements too, to keep class balance
        data_head = data_grouped.head(head_len)
        # Scaling and masks are only computed from healthy samples, so other classes are useless here
        data_head = data_head[data_head["class"] == 0]
        # Tail needs all classes for scaling
        data_tail = data_grouped.tail(
            int(
                (
                    len(train_data)
                    / (len(classes) * len(rpms))
                )
                - head_len
            )
        )

        # SCALING #
        ##
        # Robust scaling with 25 and 75 percentiles

        # Get scales
        scale = {}
        data_head_grouped = data_head.groupby(
            ["class", "rpm"], group_keys=False)
        for n, g in data_head_grouped:
            # Scale for the healthy state of each rpm
            p25 = g[sensors].quantile(0.25)
            p75 = g[sensors].quantile(0.75)
            scale[n[1]] = (p75 - p25).astype("float32") * \
                config["mixed_query_normalization_scale"]

        # Scaling helper
        def scale_group(group_data):
            group_data[sensors
                       ] = group_data[sensors] / scale[group_data.name[1]]

            return group_data

        # Scale head
        data_head = data_head_grouped.apply(scale_group)

        # Scale tail
        data_tail = data_tail.groupby(
            ["class", "rpm"], group_keys=False).apply(scale_group)

        # MASK CALCULATION #
        ##

        def create_overlapping_windows(dataframe, window_size, overlap_pct):
            num_rows = dataframe.shape[0]
            stride = int(window_size * (1 - overlap_pct))
            num_windows = (num_rows - window_size) // stride + 1

            start_indices = np.arange(num_windows) * stride
            end_indices = start_indices + window_size

            windows = [dataframe.iloc[start:end]
                       for start, end in zip(start_indices, end_indices)]

            return windows

        for n, g in data_head.groupby(["class", "rpm"]):
            windows = create_overlapping_windows(
                g, config["window_width"], config["window_overlap"])  # FIXME for synced FFT

            for sensor in sensors:
                sensor_windows = torch.tensor(
                    np.array([window[sensor].to_numpy() for window in windows]))

                fft_windows = torch.abs(torch.fft.rfft(
                    sensor_windows, norm="forward"))
                # First log, then mean should be the correct order
                if config["log_FFT"]:
                    fft_windows = torch.log1p(fft_windows)
                if not config["include_FFT_DC"]:
                    fft_windows = fft_windows[:, 1:]

                mask = fft_windows.mean(dim=0)

                masks[(n[1], sensor)] = mask

        return data_tail

    new_train_data = mixed_query_normalization_helper(train_data, "train")
    new_validation_data = mixed_query_normalization_helper(
        validation_data, "validation")
    new_test_data = mixed_query_normalization_helper(test_data, "test")

    config["masks"] = masks

    return new_train_data, new_validation_data, new_test_data


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
