import numpy as np
import torch
from scipy.signal import butter, sosfilt
from matplotlib import pyplot as plt

# Methods
#########


def lowpass_filtering(train_data, validation_data, test_data, config):
    # sos = butter(2, config["lp_filter_cutoff"], "lowpass", analog=False, output="sos", fs=3012)

    # Filtering helper
    def lowpass_filtering_helper(data, split):
        sensors = config[f"{split}_sensors"]

        def filter_group(group_data):
            # if True:
            cutoff = group_data["rpm"].iloc[0] / 60 / 3 * 50 + 30  # Up to 50x harmonics + 30 Hz
            # cutoff = group_data["rpm"].iloc[0] / 60 / 3 * 15 + 30 # Up to 15x harmonics + 30 Hz
            # print(">", group_data["rpm"].iloc[0], cutoff)
            sos = butter(4, cutoff, "lowpass", analog=False, output="sos", fs=3012)

            group_data[sensors] = sosfilt(sos, group_data[sensors].values, axis=0).astype("float32")
            # print(group_data)
            # quit()

            return group_data

        if config["data"] == "ARotor":
            data = data.groupby(["class", "rpm"], group_keys=True).apply(filter_group).reset_index(drop=True)
        elif config["data"] == "ARotor_replication":
            data = (
                data.groupby(["rpm", "torque", "severity", "fault"], group_keys=True)
                .apply(filter_group)
                .reset_index(drop=True)
            )

        return data

    new_train_data = lowpass_filtering_helper(train_data, "train")
    new_validation_data = lowpass_filtering_helper(validation_data, "validation")
    new_test_data = lowpass_filtering_helper(test_data, "test")

    return new_train_data, new_validation_data, new_test_data


def mixed_query_normalization_helper_arotor(data, config, split):
    head_len = 3012 * 6

    sensors = config[f"{split}_sensors"]

    data_grouped = data.groupby(["class", "rpm"])

    # Separate head (to be used for scaling, etc.) from measurements to be used for training/validation/testing (tail)
    # * Take the same amount away from fault measurements too, to keep class balance
    data_head = data_grouped.head(head_len)
    # Scaling and masks are only computed from healthy samples, so other classes are useless here
    data_head = data_head[data_head["class"] == 0]
    # Tail needs all classes for scaling
    data_tail = data_grouped.apply(lambda x: x.iloc[head_len:])

    # SCALING #
    ##
    # Robust scaling with 25 and 75 percentiles

    # Get scales
    scale = {}
    data_head_grouped = data_head.groupby(["class", "rpm"], group_keys=False)
    for n, g in data_head_grouped:
        # Scale for the healthy state of each rpm
        p25 = g[sensors].quantile(config["robust_scaling_low"])
        p75 = g[sensors].quantile(config["robust_scaling_high"])
        scale[n[1]] = (p75 - p25).astype("float32")  # * config["mixed_query_normalization_scale"]

    # Scaling helper
    def scale_group(group_data):
        group_data[sensors] = group_data[sensors] / scale[group_data.name[1]]

        return group_data

    # Scale head
    # ? Not really used for anything
    # data_head = data_head_grouped.apply(scale_group)

    # Scale tail
    if config["data"] == "ARotor":
        data_tail = data_tail.groupby(["class", "rpm"], group_keys=False).apply(scale_group)
    elif config["data"] == "ARotor_replication":
        data_tail = data_tail.groupby(["rpm", "torque", "severity", "fault"], group_keys=False).apply(scale_group)

    return data_tail


def mixed_query_normalization_helper_arotor_replication(data, config, split):
    head_len = 3012 * 6
    sensors = config[f"{split}_sensors"]

    data_grouped = data.groupby(["rpm", "torque", "severity", "fault"], sort=False, group_keys=False)

    # Separate head (to be used for scaling, etc.) from measurements to be used for training/validation/testing (tail)
    # * Take the same amount away from fault measurements too, to keep class balance
    data_head = data_grouped.head(head_len)
    # Scaling and masks are only computed from healthy samples, so other classes are useless here
    # In addition, only use the first GP of each split
    data_head = data_head[
        (data_head["fault"] == "baseline") & (data_head["severity"] == config[f"{split}_baseline_GPs"][0])
    ]
    # Tail needs all classes for scaling
    data_tail = data_grouped.apply(lambda x: x.iloc[head_len:])

    # SCALING #
    ##
    # Robust scaling with 25 and 75 percentiles

    # Get scales
    scale = {}
    data_head_grouped = data_head.groupby(["rpm", "torque", "severity", "fault"], group_keys=False, sort=False)
    for n, g in data_head_grouped:
        # Scale for the healthy state of each rpm
        p25 = g[sensors].quantile(config["robust_scaling_low"])
        p75 = g[sensors].quantile(config["robust_scaling_high"])
        scale[(n[0], n[1])] = (p75 - p25).astype("float32")  # * config["mixed_query_normalization_scale"]

    # Scaling helper
    def scale_group(group_data):
        group_data[sensors] = group_data[sensors] / scale[(group_data.name[0], group_data.name[1])]

        return group_data

    data_tail = data_tail.groupby(["rpm", "torque", "severity", "fault"], group_keys=False).apply(scale_group)

    return data_tail


# * Same as below, but without mask calculation
def robust_scaling(train_data, validation_data, test_data, config):
    if config["data"] == "ARotor":
        new_train_data = mixed_query_normalization_helper_arotor(train_data, config, "train")
        new_validation_data = mixed_query_normalization_helper_arotor(validation_data, config, "validation")
        new_test_data = mixed_query_normalization_helper_arotor(test_data, config, "test")
    elif config["data"] == "ARotor_replication":
        new_train_data = mixed_query_normalization_helper_arotor_replication(train_data, config, "train")
        new_validation_data = mixed_query_normalization_helper_arotor_replication(validation_data, config, "validation")
        new_test_data = mixed_query_normalization_helper_arotor_replication(test_data, config, "test")
    else:
        raise Exception("Not yet implemented for this dataset!")

    return new_train_data, new_validation_data, new_test_data


# ! DO NOT USE FOR AROTOR REPLICATION DATASET
# TODO If necessary at some point, modify for new datasets
def mixed_query_normalization(train_data, validation_data, test_data, config):
    head_len = 3012 * 10
    masks = {}
    distribution_stats = {}

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
        data_tail = data_grouped.tail(int((len(train_data) / (len(classes) * len(rpms))) - head_len))

        # SCALING #
        ##
        # Robust scaling with 25 and 75 percentiles

        # Get scales
        scale = {}
        data_head_grouped = data_head.groupby(["class", "rpm"], group_keys=False)
        for n, g in data_head_grouped:
            # Scale for the healthy state of each rpm
            p25 = g[sensors].quantile(0.25)
            p75 = g[sensors].quantile(0.75)
            scale[n[1]] = (p75 - p25).astype("float32") * config["mixed_query_normalization_scale"]

        # Scaling helper
        def scale_group(group_data):
            group_data[sensors] = group_data[sensors] / scale[group_data.name[1]]

            return group_data

        # Scale head
        data_head = data_head_grouped.apply(scale_group)

        # Scale tail
        data_tail = data_tail.groupby(["class", "rpm"], group_keys=False).apply(scale_group)

        # (1) MASK & (2) DISTRIBUTION STAT CALCULATION #
        ##

        def create_overlapping_windows(dataframe, window_size, overlap_pct):
            num_rows = dataframe.shape[0]
            stride = int(window_size * (1 - overlap_pct))
            num_windows = (num_rows - window_size) // stride + 1

            start_indices = np.arange(num_windows) * stride
            end_indices = start_indices + window_size

            windows = [dataframe.iloc[start:end] for start, end in zip(start_indices, end_indices)]

            return windows

        for n, g in data_head.groupby(["class", "rpm"]):
            windows = create_overlapping_windows(
                g, config["window_width"], config["window_overlap"]
            )  # FIXME for synced FFT

            for sensor in sensors:
                sensor_windows = torch.tensor(np.array([window[sensor].to_numpy() for window in windows]))

                fft_windows = torch.abs(torch.fft.rfft(sensor_windows, norm="forward"))
                # First log, then mean should be the correct order
                if config["log_FFT"]:
                    fft_windows = torch.log1p(fft_windows)
                if not config["include_FFT_DC"]:
                    fft_windows = fft_windows[:, 1:]

                # (2) Calculate stats to be used for std normalization

                distribution_stats[(n[1], sensor)] = {
                    "means": fft_windows.mean(dim=0),
                    "stds": fft_windows.std(dim=0, correction=1),
                }

                # (1) Create masks

                mask = fft_windows.mean(dim=0)

                masks[(n[1], sensor)] = mask

        return data_tail

    new_train_data = mixed_query_normalization_helper(train_data, "train")
    new_validation_data = mixed_query_normalization_helper(validation_data, "validation")
    new_test_data = mixed_query_normalization_helper(test_data, "test")

    config["masks"] = masks
    config["distribution_stats"] = distribution_stats

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

    if "mixed_query_normalization" in config["preprocessing_full"]:
        raise Exception("NOT CURRENTLY IN USE BECAUSE DOESN'T SUPPORT OTHER DATASETS!")
        train_data, validation_data, test_data = mixed_query_normalization(
            train_data, validation_data, test_data, config
        )
    if "lowpass_filtering" in config["preprocessing_full"]:
        train_data, validation_data, test_data = lowpass_filtering(train_data, validation_data, test_data, config)
    if "robust_scaling" in config["preprocessing_full"]:
        train_data, validation_data, test_data = robust_scaling(train_data, validation_data, test_data, config)

    return train_data, validation_data, test_data
