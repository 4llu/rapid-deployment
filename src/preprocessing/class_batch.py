import numpy as np
import torch

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


def FFT_masking(class_support_query_set, config, idx, query_samples):
    # k_shot x idx (for support samples)
    # + each query sample separately
    batch_masks = [config["masks"][rpm, sensor] for _, rpm, sensor in [
        idx for _ in range(config["k_shot"])] + query_samples]
    batch_masks = torch.stack(batch_masks)

    class_support_query_set = class_support_query_set - batch_masks

    return class_support_query_set


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
        class_support_query_set = FFT(
            class_support_query_set, config)

    if "FFT_masking" in config["preprocessing_class_batch"]:
        class_support_query_set = FFT_masking(
            class_support_query_set, config, idx, query_samples)

    return class_support_query_set
