import numpy as np
import torch
from scipy.signal import hilbert

# import matplotlib.pyplot as plt

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


def FFT(support_query_set, config, device):
    # * This changes the sample length!!!
    # * Specifically, halves it

    if config["pad_FFT"] > 0:
        padding = torch.zeros(
            *support_query_set.shape[:-1],
            config["pad_FFT"] - support_query_set.shape[-1],
            device=device if support_query_set.get_device() >= 0 else torch.device("cpu")
        )

        support_query_set = torch.cat((support_query_set, padding), dim=-1)

    # Keep only the magnitude of the positive complex conjugates
    support_query_set = torch.fft.rfft(support_query_set, norm="forward")
    support_query_set = torch.abs(support_query_set)

    # FIXME The fuck does this do?
    # if "sync_FFT" in config["preprocessing_batch"]:
    #     support_query_set = support_query_set[:, :, : config["max_fft_len"]]

    if not config["include_FFT_DC"]:
        support_query_set = support_query_set[:, :, :, 1:]

    if config["log_FFT"]:
        support_query_set = torch.log1p(support_query_set)

    if config["sample_cut"] > 0:
        support_query_set = support_query_set[:, :, :, : config["sample_cut"]]

    # fig, axs = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
    # axs = axs.flatten()
    # axs[0].plot(support_query_set[0, 0])
    # axs[1].plot(support_query_set[1, 0])
    # axs[2].plot(support_query_set[9, 0])
    # axs[3].plot(support_query_set[0, 5])
    # axs[4].plot(support_query_set[1, 5])
    # axs[5].plot(support_query_set[9, 5])

    # plt.tight_layout()
    # plt.show()
    # quit()

    return support_query_set


def FFT_w_phase(support_query_set, config, device):
    # * This changes the sample length!!!
    # * Specifically, halves it (and adds a channel)

    # Keep only the magnitude of the positive complex conjugates
    support_query_set = torch.fft.rfft(support_query_set, norm="forward")

    support_query_set = torch.cat([torch.abs(support_query_set), torch.angle(support_query_set)], dim=-2)
    # print(support_query_set.shape)

    if not config["include_FFT_DC"]:
        support_query_set = support_query_set[:, :, :, 1:]

    if config["log_FFT"]:
        support_query_set = torch.log1p(support_query_set)

    return support_query_set


def additive_white_noise(support_query_set, config, device):

    support_query_set += torch.normal(
        torch.zeros(
            support_query_set.shape, device=device if support_query_set.get_device() >= 0 else torch.device("cpu")
        ),
        config["white_noise_std"],
    )

    return support_query_set


def mult_white_noise(support_query_set, config, device):

    support_query_set *= torch.normal(
        torch.ones(
            support_query_set.shape, device=device if support_query_set.get_device() >= 0 else torch.device("cpu")
        ),
        config["white_noise_std"],
    )

    return support_query_set


def block_shuffle(support_query_set, config, device):
    original_size = support_query_set.size()
    num_blocks = support_query_set.size(-1) // config["block_size"]

    support_query_set = support_query_set.view(
        *support_query_set.size()[:-2], num_blocks, support_query_set.size()[-2], config["block_size"]
    )

    # Shuffled indices
    block_indices = torch.randperm(num_blocks)
    # Shuffle
    support_query_set = support_query_set[:, :, block_indices, :, :]

    support_query_set = support_query_set.view(original_size)

    return support_query_set


def low_freq_masking(support_query_set, config, device):
    band_width = torch.randint(low=2, high=401, size=(1,))
    support_query_set[:, :, :, :band_width] = 0

    return support_query_set


def high_freq_masking(support_query_set, config, device):
    band_width = torch.randint(low=2, high=1001, size=(1,))
    support_query_set[:, :, :, -band_width:] = 0

    return support_query_set


def random_freq_masking(support_query_set, config, device):

    num_bands = torch.randint(low=0, high=4, size=(1,))
    band_widths = torch.randint(low=2, high=401, size=(num_bands,))
    band_locations = torch.randint(low=0, high=support_query_set.shape[-1], size=(num_bands,))

    for i in range(num_bands):
        support_query_set[:, :, :, band_locations[i] : band_locations[i] + band_widths[i]] = 0

    return support_query_set


def combined_freq_masking(support_query_set, config, device):
    assert (
        sum(config["combined_freq_masking_probabilities"]) == 1
    ), "Combined probabilites of frequency masking must sum to 1!"
    assert len(config["combined_freq_masking_probabilities"]) == 4, "4 probabilities are required!"

    r = torch.rand((1,))

    # Do nothing
    if r < config["combined_freq_masking_probabilities"][0]:
        return support_query_set
    # Low masking
    if r < config["combined_freq_masking_probabilities"][0] + config["combined_freq_masking_probabilities"][1]:
        return low_freq_masking(support_query_set, config, device)
    # Low masking
    if (
        r
        < config["combined_freq_masking_probabilities"][0]
        + config["combined_freq_masking_probabilities"][1]
        + config["combined_freq_masking_probabilities"][2]
    ):
        return high_freq_masking(support_query_set, config, device)
    else:
        return random_freq_masking(support_query_set, config, device)


def gain_changer(support_query_set, config, device):
    support_query_set *= torch.normal(
        mean=torch.ones(1, device=device if support_query_set.get_device() >= 0 else torch.device("cpu")),
        std=config["gain_std"],
    )

    return support_query_set


def hilbert_envelope(support_query_set, config, device):
    mean = support_query_set.mean(dim=-1, keepdim=True)
    support_query_set = torch.tensor(np.abs(hilbert(support_query_set - mean))) + mean

    return support_query_set


# Setup
#######


def preprocess_batch(support_query_set, config, device):
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
    # ! Sync_FFT here works only if not mixing rpms
    if "FFT" in config["preprocessing_batch"]:
        support_query_set = FFT(support_query_set, config, device)
    if "sync_FFT" in config["preprocessing_batch"]:
        raise "NOT IN USE CURRENTLY!"
        # support_query_set = FFT(support_query_set, config, device)

    if "FFT_w_phase" in config["preprocessing_batch"]:
        support_query_set = FFT_w_phase(support_query_set, config, device)

    if "additive_white_noise" in config["preprocessing_batch"]:
        support_query_set = additive_white_noise(support_query_set, config, device)

    if "mult_white_noise" in config["preprocessing_batch"]:
        support_query_set = mult_white_noise(support_query_set, config, device)

    if "gain_changer" in config["preprocessing_batch"]:
        support_query_set = gain_changer(support_query_set, config, device)

    if "low_freq_masking" in config["preprocessing_batch"]:
        support_query_set = low_freq_masking(support_query_set, config, device)

    if "high_freq_masking" in config["preprocessing_batch"]:
        support_query_set = high_freq_masking(support_query_set, config, device)

    if "random_freq_masking" in config["preprocessing_batch"]:
        support_query_set = random_freq_masking(support_query_set, config, device)

    if "combined_freq_masking" in config["preprocessing_batch"]:
        support_query_set = combined_freq_masking(support_query_set, config, device)

    if "block_shuffle" in config["preprocessing_batch"]:
        support_query_set = block_shuffle(support_query_set, config, device)

    if "hilbert_envelope" in config["preprocessing_batch"]:
        support_query_set = hilbert_envelope(support_query_set, config, device)

    return support_query_set
