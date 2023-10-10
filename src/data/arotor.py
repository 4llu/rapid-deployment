import math
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from preprocessing.batch import preprocess_batch
from preprocessing.class_batch import preprocess_class_batch
from preprocessing.full import preprocess_full
from preprocessing.sample import preprocess_sample

# UTILS
#######


def fewshot_data_selection(data, sensors, rpm, classes):
    selected_data = data.loc[:, sensors + ["rpm", "class"]]
    selected_data = selected_data[(selected_data["rpm"].isin(rpm)) & (selected_data["class"].isin(classes))]

    return selected_data


class FewShotMixedDataset(Dataset):
    """
    Sample the support and query sets for one class of an episode. Combine with other sets from other classes to create
    a full episode. Compared to FewShotDataset, this dataset allows including multiple rpms or sensors in the query set.
    Support set rpm/sensor is defined by the BatchSampler, and the query set rpms/sensors are divided equally between
    the options available in the split. For non-mixed situation, possibly slightly slower than the non-mixed solution?
    """

    def __init__(self, df, config, device):
        assert not (
            config["mix_rpms"] and config["mix_sensors"]
        ), "(TODO) Setting both mix_rpms and mix_sensors as true is currently badly defined!"

        self.config = config
        self.device = device  # * Here just in case, not currently used for anything

        # Use number of unique classes as length
        self.length = len(df["class"].unique())

        # Sync FFT stuff
        self.rotation_len_map = {
            250: 2169,
            500: 1084,
            750: 723,
            1000: 542,
            1250: 434,
            1500: 361,
        }
        max_rpm = max(
            [
                *self.config["train_rpm"],
                *self.config["validation_rpm"],
                *self.config["test_rpm"],
            ]
        )
        if "sync_FFT_rotations" in self.config:
            min_window_len = self.rotation_len_map[max_rpm] * self.config["sync_FFT_rotations"]
            #! Completely new value added to config!
            self.config["max_fft_len"] = len(torch.fft.rfft(torch.arange(min_window_len)))

        # Max window width to calculate stride and max index that can be sampled
        # ! Remember to not use `self.config["window_width"]` after this point
        self.max_window_width = self.config["window_width"]
        # * If using rpm synced FFT, use the window width for the slowest rpm (longest window) as base
        if "sync_FFT" in self.config["preprocessing_sample"] or "sync_FFT" in self.config["preprocessing_batch"]:
            min_rpm = min(
                [
                    *self.config["train_rpm"],
                    *self.config["validation_rpm"],
                    *self.config["test_rpm"],
                ]
            )
            self.max_window_width = self.rotation_len_map[min_rpm] * self.config["sync_FFT_rotations"]

        # Convert overlap from % to number of time series samples
        # Done here to not calculate separately every time something is sampled
        self.window_stride = math.floor((1 - self.config["window_overlap"]) * self.max_window_width)

        # * Format data for easier sampling
        # Get the sensor columns
        sensors = list(set(df.columns) - set(["rpm", "class"]))
        # Convert to long format to include sensor in groupby
        df_long = pd.melt(df, id_vars=["rpm", "class"], value_vars=sensors, var_name="sensor")
        # Group
        self.data = df_long.groupby(["class", "rpm", "sensor"])
        # Save the shortest measurement length here because it's the easiest place
        # Divided by two to separate support and query parts
        min_measurement_length = math.floor(min(self.data.size()) / 2)

        # Convert grouped df to dict to make conversion to tensors possible
        self.data = dict(iter(self.data))
        # Convert pd Series to torch tensors
        for k in self.data.keys():
            self.data[k] = torch.tensor(self.data[k]["value"].values)

        # Calculate the number of windows per measurement
        self.max_measurement_index = math.floor((min_measurement_length - self.max_window_width) / self.window_stride)

        # Used for separating support and query sets
        self.support_offset = 0
        self.query_offset = min_measurement_length

        # Determine rpm sampling pattern for the query samples
        if self.config["mix_rpms"]:
            unique_rpms = df["rpm"].unique()
            rpm_repeats = self.config["n_query"] / len(unique_rpms)
            assert rpm_repeats % 1 == 0, "n_query needs to be divisible by the number of unique rpms in the split!"
            self.rpm_sampling_pattern = np.repeat(unique_rpms, rpm_repeats)

        # Determine sensor sampling pattern for the query samples
        # Unused if not config["mix_sensors"] == True
        if self.config["mix_sensors"]:
            sensor_repeats = self.config["n_query"] / len(sensors)
            assert sensor_repeats % 1 == 0, "n_query needs to be divisible by the number of sensors in the split!"

            self.sensor_sampling_pattern = np.repeat(sensors, sensor_repeats)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Sample support and query samples from the configured class setup. Support and
        query samples are all from the same class but the query rpm is as defined by the
        input idx and the query rpms are evenly divided between the rpms included in the
        split.

        Parameters:
            idx : tuple(int, int, string)
                tuple of class, rpm, and sensor

        Returns:
            support_query_set : tensor[support + query, 1, sample_len]
                Stacked support and query samples. dim 1 = 1 because there is currently no support for multi sensor inputs
        """

        # ! Don't use self.config["window_width"] after this point
        # * Use rotation length as window width instead if using rpm synced FFT
        window_width = self.config["window_width"]
        # Use rpm specific window_width if using synced FFT
        if "sync_FFT" in self.config["preprocessing_sample"] or "sync_FFT" in self.config["preprocessing_batch"]:
            # TODO Currently all support samples are from the same RPM
            window_width = self.rotation_len_map[idx[1]] * self.config["sync_FFT_rotations"]

        # Select random measurement windows for the support and query set
        # * 1. +1 because `.permutation` is non-inclusive of upper bound
        # * 2. Max measurement index is in terms of window index, not measurement samples,
        # * so it needs to be multiplied by the stride
        sample_idxs = (
            np.random.permutation(self.max_measurement_index + 1)[: self.config["k_shot"] + self.config["n_query"]]
            * self.window_stride
        )

        support_query_set = []

        # Support samples
        for sample_i in sample_idxs[: self.config["k_shot"]]:
            # i corresponds to window index, not measurement samples, so it need to be multiplied by the stride
            # j = i * self.window_stride # ? Done above

            sample = self.data[idx][self.support_offset + sample_i : self.support_offset + sample_i + window_width]
            support_query_set.append(sample)

        # Query samples
        # TODO Combination of the two. Requires changes to the sampling patterns defined in __init__.
        query_sampling = []
        if self.config["mix_rpms"]:
            query_sampling = zip(
                sample_idxs[self.config["k_shot"] :],  # idx
                self.rpm_sampling_pattern,  # rpm
                np.repeat(idx[2], self.config["n_query"]),  # sensor
            )
        elif self.config["mix_sensors"]:
            query_sampling = zip(
                sample_idxs[self.config["k_shot"] :],
                np.repeat(idx[1], self.config["n_query"]),
                self.sensor_sampling_pattern,
            )
        else:
            query_sampling = zip(
                sample_idxs[self.config["k_shot"] :],
                np.repeat(idx[1], self.config["n_query"]),
                np.repeat(idx[2], self.config["n_query"]),
            )
        query_sampling = list(query_sampling)

        for sample_i, sample_rpm, sample_sensor in query_sampling:
            # i corresponds to window index, not measurement samples, so it need to be multiplied by the stride
            # j = i[0] * self.window_stride # ? Done above

            # Use rpm specific window_width for each sample if using synced FFT
            if "sync_FFT" in self.config["preprocessing_sample"] or "sync_FFT" in self.config["preprocessing_batch"]:
                window_width = self.rotation_len_map[sample_rpm] * self.config["sync_FFT_rotations"]

            sample = self.data[(idx[0], sample_rpm, sample_sensor)][
                self.query_offset + sample_i : self.query_offset + sample_i + window_width
            ]
            support_query_set.append(sample)

        # Preprocess as individual samples
        support_query_set = preprocess_sample(support_query_set, self.config)

        # Combine
        support_query_set = torch.stack(support_query_set, dim=0)
        # Add channels
        support_query_set = support_query_set.unsqueeze(-2)

        # Transformations
        if len(self.config["preprocessing_class_batch"]) > 0:
            support_query_set = preprocess_class_batch(support_query_set, self.config, idx, query_sampling)

        # Swap support and query sets to mix samples between batches
        if not self.config["separate_query_and_support"]:
            self.support_offset, self.query_offset = (
                self.query_offset,
                self.support_offset,
            )

        return support_query_set, idx


class FewshotBatchSampler(Sampler):
    def __init__(self, config, classes, rpms, sensors):
        assert config["n_way"] <= len(classes), "n_way must be less than the total number of classes available!"
        self.config = config
        self.classes = classes
        self.rpms = rpms
        self.sensors = sensors

        # Generate `seq_len` at a time for efficiency
        self.seq_len = 200
        self.i = 0
        self.rpm_seq = torch.randint(high=len(self.rpms), size=(self.seq_len,))
        self.sensor_seq = torch.randint(high=len(self.sensors), size=(self.seq_len,))

        # Meant to be accessed from outside the class to find out what the last batch contained
        self.prev_batch = None

    def __len__(self):
        return math.floor(len(self.classes) / self.config["n_way"])

    def __iter__(self):

        class_perm = self.classes
        # Only permutate the class order if it matters
        if math.floor(len(self.classes) / self.config["n_way"]) != 1:
            class_perm = np.random.permutation(len(self.classes))

        # class_perm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # class_perm = [7, 0, 1, 2, 3, 4, 5, 6, 8, 9]
        # class_perm = [4, 0, 1, 2, 3, 5, 6, 7, 8, 9]

        # Go through class permutation
        for j in range(math.floor(len(self.classes) / self.config["n_way"])):
            class_batch = class_perm[j * self.config["n_way"] : (j + 1) * self.config["n_way"]]

            batch = list(
                zip(
                    class_batch,
                    [self.rpms[self.rpm_seq[self.i]] for _ in range(len(class_batch))],
                    [self.sensors[self.sensor_seq[self.i]] for _ in range(len(class_batch))],
                )
            )

            self.prev_batch = batch
            yield batch

            # Replenish rpm and sensor seqs if necessary
            self.i += 1
            if self.i == self.seq_len:
                self.i = 0
                self.rpm_seq = torch.randint(high=len(self.rpms), size=(self.seq_len,))
                self.sensor_seq = torch.randint(high=len(self.sensors), size=(self.seq_len,))


def fewshot_collate(batch, config=None, device=None):
    """
    Replacement for default collate_fn to deal with the labels being tuples.

    Parameters:
        See PyTorch default collate_fn

    Returns:
        samples: torch.tensor[classes, k_shot + n_query, window_length]
        labels: list(tuple(int, int, string) x len(classes))
            Each label is a tuple containing class, rpm, and sensor
    """
    # Realign dataset `__getitem__` outputs
    samples, labels = list(zip(*batch))
    samples = torch.stack(samples)

    samples = preprocess_batch(samples, config, device)

    # * For all other devices except mps, move to device here,
    # * because most preprocessing operations are faster on GPU.
    # * However, some operations (e.g. FFT) don't work on MPS yet,
    # * so operate on cpu here
    # if str(device) != "mps":
    #     # Non-blocking probably has no effect here,
    #     # because model(samples) is an immediate sync point
    #     samples = samples.to(device, non_blocking=True)

    # if len(config["preprocessing_batch"]) > 0:
    #     samples = preprocess_batch(samples, config, device)

    # # Move samples to MPS
    # if str(device) == "mps":
    #     # Non-blocking probably has no effect here,
    #     # because model(samples) is an immediate sync point
    #     samples = samples.to(device, non_blocking=True)

    return samples, labels


# DATA DIVISIONS
################


def get_arotor_data(config, device):
    """
    Create train/val/test dataloaders from the ARotor dataset according to the given configuration.

    Parameters:
        config: dict
            The chosen configuration dict from /configs.
        device: torch.device
            Not currently used for anything. Can be used to move all data to
            GPU at start in the future.

    Returns:
        train_loader: torch.Dataloader
        validation_loader: torch.Dataloader
        test_loader: torch.Dataloader
    """
    # Load data
    ###########

    print("Reading data")

    abs_path = os.path.dirname(__file__)
    data_folder = os.path.join(abs_path, os.pardir, os.pardir, "data")

    # XXX HUOM! V2 version in use
    data = pd.read_feather(os.path.join(data_folder, "processed", "arotor_V2.feather"))
    # data = pd.read_feather(os.path.join(data_folder, "processed", "arotor_enc_angle_02_res_PCHIP_V2.feather"))
    # data = pd.read_feather(os.path.join(data_folder, "processed", "arotor.feather"))

    # Data selection
    ################

    print("Formatting data")

    train_data = fewshot_data_selection(data, config["train_sensors"], config["train_rpm"], config["train_classes"])
    validation_data = fewshot_data_selection(
        data,
        config["validation_sensors"],
        config["validation_rpm"],
        config["validation_classes"],
    )
    test_data = fewshot_data_selection(data, config["test_sensors"], config["test_rpm"], config["test_classes"])

    # Data preprocessing for full measurements
    ##########################################

    train_data, validation_data, test_data = preprocess_full(train_data, validation_data, test_data, config)

    # Dataset creation
    ##################

    train_dataset = FewShotMixedDataset(train_data, config, device)
    validation_dataset = FewShotMixedDataset(validation_data, config, device)
    test_dataset = FewShotMixedDataset(test_data, config, device)

    # Dataloaders
    #############

    # Create samplers
    train_sampler = FewshotBatchSampler(
        config,
        config["train_classes"],
        config["train_rpm"],
        config["train_sensors"],
    )
    validation_sampler = FewshotBatchSampler(
        config,
        config["validation_classes"],
        config["validation_rpm"],
        config["validation_sensors"],
    )
    test_sampler = FewshotBatchSampler(
        config,
        config["test_classes"],
        config["test_rpm"],
        config["test_sensors"],
    )

    # Create dataloaders
    # TODO Test other num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=partial(fewshot_collate, config=config, device=device),
        pin_memory=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=partial(fewshot_collate, config=config, device=device),
        pin_memory=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=partial(fewshot_collate, config=config, device=device),
        pin_memory=False,
        num_workers=0,
    )

    return train_loader, validation_loader, test_loader
