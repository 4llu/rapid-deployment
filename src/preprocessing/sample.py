import torch


def sync_FFT(sample_list, config):
    """
    Use this if batches include mixed rpms. Otherwise prefer batch level operations.
    """
    new_sample_list = []

    for s in sample_list:
        s_new = torch.fft.rfft(s, norm="forward")
        s_new = torch.abs(s_new)
        s_new = s_new[: config["max_fft_len"]]

        if not config["include_FFT_DC"]:
            s_new = s_new[1:]

        new_sample_list.append(s_new)

    return new_sample_list


def FFT(sample_list, config):
    """
    For testing purposes. Never actually use this.
    """
    new_sample_list = []
    for s in sample_list:
        s_new = torch.fft.rfft(s, norm="forward")
        s_new = torch.abs(s_new)

        if not config["include_FFT_DC"]:
            s_new = s_new[1:]

        new_sample_list.append(s_new)

    return new_sample_list


def preprocess_sample(sample_list, config):
    """
    Preprocess a sample during runtime. Note that this means two things:

    1. The inputs are samples, i.e. measurement windows, not full measurements.
    2. The results are not saved, so the transformation are repeated every epoch.
    3. !!! Input samples can be of different lengths, but output samples should alway be the same length

    Technically you can do all of these, but most of the time only some combinations make sense
    (or are even possible). Also, the order they are in here matters if some normalizations are
    meant to be run together.
    """

    # RPM synced FFT
    if "sync_FFT" in config["preprocessing_sample"]:
        raise " NOT IN USE CURRENTLY!"
        # sample_list = sync_FFT(sample_list, config)
    if "FFT" in config["preprocessing_sample"]:
        raise "SLOW! ARE YOU SURE YOU WANT TO USE THIS? IN THAT CASE COMMENT THIS `raise`"
        # sample_list = FFT(sample_list, config)

    return sample_list
