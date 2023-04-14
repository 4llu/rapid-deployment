import math

import numpy as np
import scipy
import torch
import torchaudio

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


def min_max_helper(batch, min, max):
    return (batch - min) / (max - min)


def individual_min_max(train_data, validation_data, test_data, config):
    train_data = min_max_helper(
        train_data,
        train_data.min(dim=-1, keepdim=True)[0],
        train_data.max(dim=-1, keepdim=True)[0],
    )

    validation_data = min_max_helper(
        validation_data,
        validation_data.min(dim=-1, keepdim=True)[0],
        validation_data.max(dim=-1, keepdim=True)[0],
    )

    test_data = min_max_helper(
        test_data,
        test_data.min(dim=-1, keepdim=True)[0],
        test_data.max(dim=-1, keepdim=True)[0],
    )

    return train_data, validation_data, test_data


def FFT(train_data, validation_data, test_data, config):
    # * This changes the sample length!!!
    # * Specifically, halves it

    # Keep only the magnitude of the positive complex conjugates
    train_data = torch.fft.rfft(train_data)
    train_data = torch.abs(train_data)

    validation_data = torch.fft.rfft(validation_data)
    validation_data = torch.abs(validation_data)

    test_data = torch.fft.rfft(test_data)
    test_data = torch.abs(test_data)

    return train_data, validation_data, test_data


def lpf(samples, cutoff_filter):
    # * Helper for below

    return torch.real(torch.fft.ifft(torch.fft.fft(samples) * cutoff_filter))


def low_pass_filter(train_data, validation_data, test_data, config):
    # * https://stackoverflow.com/questions/70825086/python-lowpass-filter-with-only-numpy
    # * https://colab.research.google.com/drive/1RR_9EYlApDMg4jAS2HuJIpSqwg5RLzGW?usp=sharing#scrollTo=UCuE2tG3I73h

    cutoff_freq = config["low_pass_filter_cutoff"]  # Hz
    sampling_freq = 10000
    cutoff_filter = (
        1.0 * torch.abs(torch.fft.fftfreq(train_data.size(-1), 1.0 / sampling_freq))
        <= cutoff_freq
    )

    train_data = lpf(train_data, cutoff_filter)
    validation_data = lpf(validation_data, cutoff_filter)
    test_data = lpf(test_data, cutoff_filter)

    return train_data, validation_data, test_data


def ra(samples, pad_sizes, ra_window):
    # Through numpy
    samples = samples.numpy()
    # Pad
    samples = np.concatenate(
        [samples[:, :, : pad_sizes[0]], samples, samples[:, :, -pad_sizes[1] :]],
        axis=-1,
    )
    # Convolve
    samples = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones([ra_window]) / ra_window, mode="valid"),
        axis=-1,
        arr=samples,
    )

    return torch.tensor(samples, dtype=torch.float32)


def rolling_average(train_data, validation_data, test_data, config):

    pad_len = (config["rolling_average_window"] - 1) / 2
    left_pad = math.floor(pad_len)
    right_pad = math.ceil(pad_len)
    pad_sizes = (left_pad, right_pad)

    train_data = ra(train_data, pad_sizes, config["rolling_average_window"])
    validation_data = ra(validation_data, pad_sizes, config["rolling_average_window"])
    test_data = ra(test_data, pad_sizes, config["rolling_average_window"])

    return train_data, validation_data, test_data


def subsample(train_data, validation_data, test_data, config):

    train_data = train_data[:, :, :: config["subsample_step"]]
    validation_data = validation_data[:, :, :: config["subsample_step"]]
    test_data = test_data[:, :, :: config["subsample_step"]]

    return train_data, validation_data, test_data


# MAIN
######


def preprocessing(train_data, validation_data, test_data, config):
    # * Technically you can do all of these, but most of the time only some combinations make sense
    # * (or are even possible). Also, the order they are in here matters if some normalizations are
    # * meant to be run together

    # Z-score
    if "z-score" in config["preprocessing"]:
        train_data, validation_data, test_data = z_score(
            train_data, validation_data, test_data, config
        )

    # Min max
    if "min_max" in config["preprocessing"]:
        train_data, validation_data, test_data = min_max(
            train_data, validation_data, test_data, config
        )

    # Individual min max
    if "individual_min_max" in config["preprocessing"]:
        train_data, validation_data, test_data = individual_min_max(
            train_data, validation_data, test_data, config
        )

    # Low pass filter
    if "low_pass_filter" in config["preprocessing"]:
        train_data, validation_data, test_data = low_pass_filter(
            train_data, validation_data, test_data, config
        )

    # Rolling average
    if "rolling_average" in config["preprocessing"]:
        train_data, validation_data, test_data = rolling_average(
            train_data, validation_data, test_data, config
        )

    # Subsampling
    if "subsample" in config["preprocessing"]:
        train_data, validation_data, test_data = subsample(
            train_data, validation_data, test_data, config
        )

    # FFT
    if "FFT" in config["preprocessing"]:
        train_data, validation_data, test_data = FFT(
            train_data, validation_data, test_data, config
        )

    return train_data, validation_data, test_data
