import torch


# Methods
#########


def min_max_helper(batch, min, max):
    return (batch - min) / (max - min)


def individual_min_max(support_query_set, config):
    #! FIXME Check that this is working correctly

    support_query_set = min_max_helper(
        support_query_set,
        support_query_set.min(dim=-1, keepdim=True)[0],
        support_query_set.max(dim=-1, keepdim=True)[0],
    )

    return support_query_set


def FFT(support_query_set, config):
    #! FIXME Check that this is working correctly
    # * This changes the sample length!!!
    # * Specifically, halves it

    # Keep only the magnitude of the positive complex conjugates
    print(support_query_set.shape)
    print(support_query_set.dtype)
    print(support_query_set.device)
    support_query_set = torch.fft.rfft(support_query_set)
    support_query_set = torch.abs(support_query_set)
    print(support_query_set.shape)
    print(support_query_set.dtype)
    print(support_query_set.device)
    quit()

    return support_query_set


# Setup
#######


def preprocess_batch(support_query_set, config):
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
    if "individual_min_max" in config["preprocessing_batch"]:
        support_query_set = individual_min_max(support_query_set, config)

    # FFT
    if "FFT" in config["preprocessing_batch"]:
        support_query_set = FFT(support_query_set, config)

    return support_query_set
