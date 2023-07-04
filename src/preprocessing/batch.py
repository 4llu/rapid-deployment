import torch
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


def FFT(support_query_set, config):
    # * This changes the sample length!!!
    # * Specifically, halves it

    # Keep only the magnitude of the positive complex conjugates
    support_query_set = torch.fft.rfft(support_query_set, norm="forward")
    support_query_set = torch.abs(support_query_set)

    # FIXME The fuck does this do?
    if "sync_FFT" in config["preprocessing_batch"]:
        support_query_set = support_query_set[:, :, : config["max_fft_len"]]

    if not config["include_FFT_DC"]:
        support_query_set = support_query_set[:, :, 1:]

    if config["log_FFT"]:
        support_query_set = torch.log1p(support_query_set)

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


def additive_white_noise(support_query_set, config):

    support_query_set += torch.normal(torch.zeros(support_query_set.shape), torch.ones(
        support_query_set.shape) * config["white_noise_std"])

    return support_query_set


def mult_white_noise(support_query_set, config):

    support_query_set *= 1 + torch.normal(torch.zeros(support_query_set.shape), torch.ones(
        support_query_set.shape) * config["white_noise_std"])

    return support_query_set


def block_shuffle(support_query_set, config):
    # print(support_query_set.size())
    # print(support_query_set[0, 0, :200].mean())

    original_size = support_query_set.size()
    num_blocks = support_query_set.size(-1) // config["block_size"]

    support_query_set = support_query_set.view(
        *support_query_set.size()[:-1], num_blocks, config["block_size"])

    # Shuffled indices
    block_indices = torch.randperm(num_blocks)
    # Shuffle
    support_query_set = support_query_set[:, :, block_indices, :]

    support_query_set = support_query_set.view(original_size)

    # print(support_query_set[0, 0, :200].mean())
    # print(support_query_set.size())
    # quit()

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
        support_query_set = individual_min_max(
            support_query_set, config)

    # FFT
    # ! Sync_FFT here works only if not mixing rpms
    if (
        "FFT" in config["preprocessing_batch"]
        or "sync_FFT" in config["preprocessing_batch"]
    ):
        support_query_set = FFT(support_query_set, config)

    if "additive_white_noise" in config["preprocessing_batch"]:
        support_query_set = additive_white_noise(support_query_set, config)

    if "mult_white_noise" in config["preprocessing_batch"]:
        support_query_set = mult_white_noise(support_query_set, config)

    if "block_shuffle" in config["preprocessing_batch"]:
        support_query_set = block_shuffle(support_query_set, config)

    return support_query_set
