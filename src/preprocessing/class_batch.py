import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from matplotlib import pyplot as plt

# Methods
#########


def FFT(support_query_set, config):
    # * This changes the sample length!!!
    # * Specifically, halves it

    # Keep only the magnitude of the positive complex conjugates
    support_query_set = torch.fft.rfft(support_query_set, norm="forward")
    support_query_set = torch.abs(support_query_set)

    if not config["include_FFT_DC"]:
        support_query_set = support_query_set[:, :, 1:]

    if config["log_FFT"]:
        support_query_set = torch.log1p(support_query_set)

    return support_query_set


def FFT_mean_std_channels(class_support_query_set, config, idx, query_samples):
    batch_means = [
        config["distribution_stats"][rpm, sensor]["means"]
        for _, rpm, sensor in [idx for _ in range(config["k_shot"])] + query_samples
    ]
    # Masks don't have channels. so they need to be added separately
    batch_means = torch.stack(batch_means).unsqueeze(-2)

    batch_stds = [
        config["distribution_stats"][rpm, sensor]["stds"]
        for _, rpm, sensor in [idx for _ in range(config["k_shot"])] + query_samples
    ]
    # Masks don't have channels. so they need to be added separately
    batch_stds = torch.stack(batch_stds).unsqueeze(-2)

    class_support_query_set = torch.cat([class_support_query_set, batch_means, batch_stds], dim=1)

    return class_support_query_set


def FFT_masking(class_support_query_set, config, idx, query_samples):
    # k_shot x idx (for support samples)
    # + each query sample separately
    batch_masks = [
        config["masks"][rpm, sensor] for _, rpm, sensor in [idx for _ in range(config["k_shot"])] + query_samples
    ]
    # Masks don't have channels. so they need to be added separately
    batch_masks = torch.stack(batch_masks).unsqueeze(-2)

    class_support_query_set = class_support_query_set - batch_masks

    return class_support_query_set


def FFT_std_normalization(class_support_query_set, config, idx, query_samples):

    batch_means = [
        config["distribution_stats"][rpm, sensor]["means"]
        for _, rpm, sensor in [idx for _ in range(config["k_shot"])] + query_samples
    ]
    # Masks don't have channels. so they need to be added separately
    batch_means = torch.stack(batch_means).unsqueeze(-2)

    batch_stds = [
        config["distribution_stats"][rpm, sensor]["stds"]
        for _, rpm, sensor in [idx for _ in range(config["k_shot"])] + query_samples
    ]
    # Masks don't have channels. so they need to be added separately
    batch_stds = torch.stack(batch_stds).unsqueeze(-2)

    new_class_support_query_set = (class_support_query_set - batch_means) / batch_stds

    # print(batch_means.shape)
    # print(batch_stds.shape)
    # print(class_support_query_set.shape)
    # quit()

    # fig, axs = plt.subplots(3, 2, figsize=(20, 12), sharey="col")
    # axs = axs.flatten()
    # bin_width = 3012 / 6000
    # trunc = 300
    # x = np.arange(0, 3000) * bin_width + bin_width/2
    # axs[0].plot(x[:trunc], class_support_query_set[0][:trunc])
    # axs[0].set_title(str(idx))
    # axs[1].plot(x[:trunc], new_class_support_query_set[0][:trunc])
    # axs[1].set_title(str(idx))

    # axs[2].plot(x[:trunc], class_support_query_set[1][:trunc])
    # axs[2].set_title(str(query_samples[0]))
    # axs[3].plot(x[:trunc], new_class_support_query_set[1][:trunc])
    # axs[3].set_title(str(query_samples[0]))

    # axs[4].plot(x[:trunc], class_support_query_set[-1][:trunc])
    # axs[4].set_title(str(query_samples[-1]))
    # axs[5].plot(x[:trunc], new_class_support_query_set[-1][:trunc])
    # axs[5].set_title(str(query_samples[-1]))

    # plt.tight_layout()
    # plt.show()
    # quit()

    return new_class_support_query_set


rpm_rotation_lengths = {
    250: 2164,
    500: 1082,
    750: 721,
    1000: 541,
    1250: 433,
    1500: 361,
}


def length_increase_interpolation(class_support_query_set, config, idx, query_samples):
    new_class_support_query_set = []

    truncated_support_set = class_support_query_set[
        : config["k_shot"], :, : rpm_rotation_lengths[idx[1]] * config["interpolated_rotations"]
    ]
    if idx[1] == 250:
        new_class_support_query_set.extend(truncated_support_set)
    else:
        new_support_x = np.linspace(
            0,
            rpm_rotation_lengths[idx[1]] * config["interpolated_rotations"] - 1,
            rpm_rotation_lengths[250] * config["interpolated_rotations"],
            endpoint=True,
        )
        new_support_set = Akima1DInterpolator(range(truncated_support_set.shape[-1]), truncated_support_set, axis=2)(
            new_support_x
        )
        new_class_support_query_set.extend(new_support_set)

    # Count number of unique rpms
    unique_rpms = np.unique(np.array(query_samples)[:, 1])
    unique_rpms = [int(x) for x in unique_rpms]
    query_rpm_set_len = int((class_support_query_set.shape[0] - config["k_shot"]) / len(unique_rpms))

    for i in range(len(unique_rpms)):
        truncated_query_rpm = class_support_query_set[
            config["k_shot"] + i * query_rpm_set_len : config["k_shot"] + (i + 1) * query_rpm_set_len,
            :,
            : rpm_rotation_lengths[unique_rpms[i]] * config["interpolated_rotations"],
        ]
        if unique_rpms[i] == 250:
            new_class_support_query_set.extend(truncated_query_rpm)
        else:
            new_x = np.linspace(
                0,
                rpm_rotation_lengths[unique_rpms[i]] * config["interpolated_rotations"] - 1,
                rpm_rotation_lengths[250] * config["interpolated_rotations"],
                endpoint=True,
            )
            new_query_rpm = Akima1DInterpolator(range(truncated_query_rpm.shape[-1]), truncated_query_rpm, axis=2)(
                new_x
            )
            new_class_support_query_set.extend(new_query_rpm)

    new = torch.tensor(np.stack(new_class_support_query_set, axis=0), dtype=torch.float32)

    return new


def TSA(class_support_query_set, config, idx, query_samples):
    new_class_support_query_set = []

    # Remove unnecessary
    truncated_support_set = class_support_query_set[
        : config["k_shot"], :, : rpm_rotation_lengths[idx[1]] * config["TSA_rotations"] * config["TSA_cycles"]
    ]
    # TSA
    new_support_set = truncated_support_set.reshape(
        truncated_support_set.shape[0],
        truncated_support_set.shape[1],
        config["TSA_cycles"],
        rpm_rotation_lengths[idx[1]] * config["TSA_rotations"],
    )
    new_support_set = torch.mean(new_support_set, axis=-2)

    # Interpolate
    if idx[1] != 250:
        new_support_x = np.linspace(
            0,
            rpm_rotation_lengths[idx[1]] * config["TSA_rotations"] - 1,
            rpm_rotation_lengths[250] * config["TSA_rotations"],
            endpoint=True,
        )
        new_support_set = Akima1DInterpolator(range(new_support_set.shape[-1]), new_support_set, axis=-1)(new_support_x)

    # Save
    new_class_support_query_set.extend(new_support_set)

    # Count number of unique rpms
    unique_rpms = np.unique(np.array(query_samples)[:, 1])
    unique_rpms = [int(x) for x in unique_rpms]
    query_rpm_set_len = int((class_support_query_set.shape[0] - config["k_shot"]) / len(unique_rpms))

    for i in range(len(unique_rpms)):
        # Remove unnecessary
        truncated_query_rpm = class_support_query_set[
            config["k_shot"] + i * query_rpm_set_len : config["k_shot"] + (i + 1) * query_rpm_set_len,
            :,
            : rpm_rotation_lengths[unique_rpms[i]] * config["TSA_rotations"] * config["TSA_cycles"],
        ]

        # TSA
        new_query_rpm = truncated_query_rpm.reshape(
            truncated_query_rpm.shape[0],
            truncated_query_rpm.shape[1],
            config["TSA_cycles"],
            rpm_rotation_lengths[unique_rpms[i]] * config["TSA_rotations"],
        )
        new_query_rpm = torch.mean(new_query_rpm, axis=-2)

        # Interpolate
        if unique_rpms[i] != 250:
            new_query_x = np.linspace(
                0,
                rpm_rotation_lengths[unique_rpms[i]] * config["TSA_rotations"] - 1,
                rpm_rotation_lengths[250] * config["TSA_rotations"],
                endpoint=True,
            )
            new_query_rpm = Akima1DInterpolator(range(new_query_rpm.shape[-1]), new_query_rpm, axis=-1)(new_query_x)

        # Save
        new_class_support_query_set.extend(new_query_rpm)

    new = torch.tensor(np.stack(new_class_support_query_set, axis=0), dtype=torch.float32)

    return new


# Setup
#######


def preprocess_class_batch(class_support_query_set, config, idx, query_samples):
    # FIXME When should this be run?
    """
    Preprocess a batch during runtime. Note that this means two things:

    1. The inputs are samples, i.e. measurement windows, not full measurements.
    2. The results are not saved, so the transformation are repeated every epoch.

    Technically you can do all of these, but most of the time only some combinations make sense
    (or are even possible). Also, the order they are in here matters if some normalizations are
    meant to be run together.
    """

    # Individual min max
    if "TSA" in config["preprocessing_class_batch"]:
        class_support_query_set = TSA(class_support_query_set, config, idx, query_samples)

    if "FFT" in config["preprocessing_class_batch"]:
        class_support_query_set = FFT(class_support_query_set, config)

    if "FFT_mean_std_channels" in config["preprocessing_class_batch"]:
        class_support_query_set = FFT_mean_std_channels(class_support_query_set, config, idx, query_samples)

    if "FFT_masking" in config["preprocessing_class_batch"]:
        class_support_query_set = FFT_masking(class_support_query_set, config, idx, query_samples)

    if "FFT_std_normalization" in config["preprocessing_class_batch"]:
        class_support_query_set = FFT_std_normalization(class_support_query_set, config, idx, query_samples)
    if "length_increase_interpolation" in config["preprocessing_class_batch"]:
        class_support_query_set = length_increase_interpolation(class_support_query_set, config, idx, query_samples)

    return class_support_query_set
