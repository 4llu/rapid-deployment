import numpy as np
import torch
import matplotlib.pyplot as plt

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
    if "FFT" in config["preprocessing_class_batch"]:
        class_support_query_set = FFT(class_support_query_set, config)

    if "FFT_mean_std_channels" in config["preprocessing_class_batch"]:
        class_support_query_set = FFT_mean_std_channels(class_support_query_set, config, idx, query_samples)

    if "FFT_masking" in config["preprocessing_class_batch"]:
        class_support_query_set = FFT_masking(class_support_query_set, config, idx, query_samples)

    if "FFT_std_normalization" in config["preprocessing_class_batch"]:
        class_support_query_set = FFT_std_normalization(class_support_query_set, config, idx, query_samples)

    return class_support_query_set
