import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose

from data.augmentation import AugmentationCollation, WhiteNoise
from data.preprocessing import preprocessing

# UTILS
#######


def format_df_sample(g, columns):
    # Crop to 1600 samples
    # Some samples are 1601-1603 long
    sample = torch.tensor(g[columns].values, dtype=torch.float).T[:, :1600]
    # Pad samples under 1600 long by repeating the last value
    # Some samples are 1599 long
    # FIXME Won't work for samples under 1599 long
    if sample.shape[1] < 1600:
        sample = torch.cat([sample, sample[:, -1:]], dim=1)

    return sample


# DATA DIVISIONS
################


def get_simulated_data(config):
    abs_path = os.path.dirname(__file__)
    data_folder = os.path.join(abs_path, os.pardir, os.pardir, "data")

    # Load data
    raw_data = pd.read_feather(
        os.path.join(data_folder, "processed", "simulated.feather")
    )
    # Divide data by measurement cycles
    class_data = raw_data.groupby(["class", "repetition"])

    # Train/validation/test split
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    test_data = []
    test_labels = []

    # 50/25/25
    # * Based on the assumption that there are 100 repetitions of each class
    for n, g in class_data:
        if n[1] <= 50:
            train_labels.append(n[0])
            train_data.append(format_df_sample(g, config["signals"]))
        elif n[1] <= 75:
            validation_labels.append(n[0])
            validation_data.append(format_df_sample(g, config["signals"]))
        else:
            test_labels.append(n[0])
            test_data.append(format_df_sample(g, config["signals"]))

    # Make data and labels fully into tensors
    train_data = torch.stack(train_data)
    validation_data = torch.stack(validation_data)
    test_data = torch.stack(test_data)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    test_labels = torch.tensor(test_labels)

    return (
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
    )


# def get_sim2real_data(config):
#     abs_path = os.path.dirname(__file__)
#     data_folder = os.path.join(abs_path, os.pardir, os.pardir, "data")

#     # Load data
#     raw_simulated_data = pd.read_feather(
#         os.path.join(data_folder, "processed", "simulated.feather")
#     )
#     raw_measured_data = pd.read_feather(
#         os.path.join(data_folder, "processed", "measured.feather")
#     )
#     # Divide data by measurement cycles
#     class_simulated_data = raw_simulated_data.groupby(["class", "repetition"])
#     class_measured_data = raw_measured_data.groupby(["class", "repetition"])

#     # Train/validation/test split
#     train_data = []
#     train_labels = []
#     validation_data = []
#     validation_labels = []
#     test_data = []
#     test_labels = []

#     # 75/25 -> training/validation split for simulated data
#     # * Based on the assumption that there are 100 repetitions of each class
#     for n, g in class_simulated_data:
#         sample = format_df_sample(g, config["signals"])

#         # * Offset preprocessing is done here so that which samples are
#         # * measured and which are sample is still known
#         if config["offset_size"] > 0:
#             # Simulation samples are only shortened
#             sample = sample[:, : -config["offset_size"]]

#         if n[1] <= 75:
#             train_labels.append(n[0])
#             train_data.append(sample)
#         else:
#             validation_labels.append(n[0])
#             validation_data.append(sample)

#     # 100% measured data used for testing
#     for n, g in class_measured_data:
#         sample = format_df_sample(g, config["signals"])

#         # * Offset preprocessing is done here so that which samples are
#         # * measured and which are sample is still known
#         if config["offset_size"] > 0:
#             # Measured samples are offset forward
#             sample = sample[:, config["offset_size"] :]

#         test_labels.append(n[0])
#         test_data.append(sample)

#     # print(len(train_data))
#     # print(len(validation_data))
#     # print(len(test_data))

#     # Make data and labels fully into tensors
#     train_data = torch.stack(train_data)
#     validation_data = torch.stack(validation_data)
#     test_data = torch.stack(test_data)

#     train_labels = torch.tensor(train_labels)
#     validation_labels = torch.tensor(validation_labels)
#     test_labels = torch.tensor(test_labels)

#     return (
#         train_data,
#         train_labels,
#         validation_data,
#         validation_labels,
#         test_data,
#         test_labels,
#     )


def get_sim2real_data(config):
    # * Copy paste from get_sim2real_data with the exception of the data split

    abs_path = os.path.dirname(__file__)
    data_folder = os.path.join(abs_path, os.pardir, os.pardir, "data")

    # Load data
    raw_simulated_data = pd.read_feather(
        os.path.join(data_folder, "processed", "simulated.feather")
    )
    raw_measured_data = pd.read_feather(
        os.path.join(data_folder, "processed", "measured.feather")
    )
    # Divide data by measurement cycles
    class_simulated_data = raw_simulated_data.groupby(["class", "repetition"])
    class_measured_data = raw_measured_data.groupby(["class", "repetition"])

    # Train/validation/test split
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    test_data = []
    test_labels = []

    # 75/25/0 -> training/validation/test split for simulated data
    # * Based on the assumption that there are 100 repetitions of each class
    for n, g in class_simulated_data:
        sample = format_df_sample(g, config["signals"])

        # * Offset preprocessing is done here so that which samples are
        # * measured and which are sample is still known
        if config["offset_size"] > 0:
            # Simulation samples are only shortened
            sample = sample[:, : -config["offset_size"]]

        if n[1] <= 75:
            train_labels.append(n[0])
            train_data.append(sample)
        else:
            validation_labels.append(n[0])
            validation_data.append(sample)

    # print("Simulated", len(validation_data))
    # * The validation set will contain a mix of simulated and measured data
    for n, g in class_measured_data:
        sample = format_df_sample(g, config["signals"])

        # Offset sample if configured
        if config["offset_size"] > 0:
            # Measured samples are offset forward
            sample = sample[:, config["offset_size"] :]

        # Mix a few measured saples into validation
        if n[1] <= 10:
            # validation_labels.append(n[0])
            # validation_data.append(sample)
            pass
        else:
            test_labels.append(n[0])
            test_data.append(sample)

    # print("Simulated + measured", len(validation_data))
    # un, co = np.unique(np.array(validation_labels), return_counts=True)
    # print("unique", list(zip(un, co)))
    # quit()

    # Make data and labels fully into tensors
    train_data = torch.stack(train_data)
    validation_data = torch.stack(validation_data)
    test_data = torch.stack(test_data)

    print(train_data.shape)
    print(validation_data.shape)
    print(test_data.shape)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    test_labels = torch.tensor(test_labels)

    return (
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
    )


def get_sim2real_h_opt_data(config):
    # * Copy paste from get_sim2real_data with the exception of the data split

    abs_path = os.path.dirname(__file__)
    data_folder = os.path.join(abs_path, os.pardir, os.pardir, "data")

    # Load data
    raw_simulated_data = pd.read_feather(
        os.path.join(data_folder, "processed", "simulated.feather")
    )
    raw_measured_data = pd.read_feather(
        os.path.join(data_folder, "processed", "measured.feather")
    )
    # Divide data by measurement cycles
    class_simulated_data = raw_simulated_data.groupby(["class", "repetition"])
    class_measured_data = raw_measured_data.groupby(["class", "repetition"])

    # Train/validation/test split
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    test_data = []
    test_labels = []

    # 75/25/0 -> training/validation/test split for simulated data
    # * Based on the assumption that there are 100 repetitions of each class
    for n, g in class_simulated_data:
        sample = format_df_sample(g, config["signals"])

        # * Offset preprocessing is done here so that which samples are
        # * measured and which are sample is still known
        if config["offset_size"] > 0:
            # Simulation samples are only shortened
            sample = sample[:, : -config["offset_size"]]

        if n[1] <= 75:
            train_labels.append(n[0])
            train_data.append(sample)
        else:
            validation_labels.append(n[0])
            validation_data.append(sample)

    # print("Simulated", len(validation_data))
    # * The validation set will contain a mix of simulated and measured data
    for n, g in class_measured_data:
        sample = format_df_sample(g, config["signals"])

        # Offset sample if configured
        if config["offset_size"] > 0:
            # Measured samples are offset forward
            sample = sample[:, config["offset_size"] :]

        # Mix a few measured saples into validation
        if n[1] <= 10:
            validation_labels.append(n[0])
            validation_data.append(sample)
        else:
            test_labels.append(n[0])
            test_data.append(sample)

    # print("Simulated + measured", len(validation_data))
    # un, co = np.unique(np.array(validation_labels), return_counts=True)
    # print("unique", list(zip(un, co)))
    # quit()

    # Make data and labels fully into tensors
    train_data = torch.stack(train_data)
    validation_data = torch.stack(validation_data)
    test_data = torch.stack(test_data)

    print(train_data.shape)
    print(validation_data.shape)
    print(test_data.shape)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    test_labels = torch.tensor(test_labels)

    return (
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
    )


# SETUP
#######


def setup_data(config):
    print("PREPARING DATA")

    # DATA INITIALIZATION
    #####################
    if config["data"] == "simulated":
        (
            train_data,
            train_labels,
            validation_data,
            validation_labels,
            test_data,
            test_labels,
        ) = get_simulated_data(config)
    elif config["data"] == "sim2real":
        (
            train_data,
            train_labels,
            validation_data,
            validation_labels,
            test_data,
            test_labels,
        ) = get_sim2real_data(config)
    elif config["data"] == "sim2real_h_opt":
        (
            train_data,
            train_labels,
            validation_data,
            validation_labels,
            test_data,
            test_labels,
        ) = get_sim2real_h_opt_data(config)
    else:
        raise Exception("No such data configuration as`", config["data"], "`!")

    # PREPROCESSING
    ###############

    train_data, validation_data, test_data = preprocessing(
        train_data, validation_data, test_data, config
    )

    # DATASET
    #########

    # Dataset conversion
    train_dataset = TensorDataset(
        train_data, train_labels
    )  # Only apply transforms to training set
    validation_dataset = TensorDataset(validation_data, validation_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # AUGMENTATION
    ##############

    transforms = []

    if "white_noise" in config["augmentation"]:
        transforms.append(
            WhiteNoise(config["white_noise_std"], config["white_noise_spacing"])
        )

    if len(transforms) > 0:
        transforms = Compose(transforms)
    else:
        transforms = None

    augmentor = AugmentationCollation(transforms)

    # DATALOADER
    ############

    # * Seems like the data is small enough that adding num_workers slows down the loading process
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        # num_workers=4,
        collate_fn=augmentor,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        # num_workers=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        # num_workers=4,
    )

    print("PREPARING DATA DONE")

    return train_dataloader, validation_dataloader, test_dataloader
