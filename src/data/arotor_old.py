import math
import os
from numpy.random import random_sample, randint

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class FewShotDataset(Dataset):
    def __init__(self, data, config, device):
        self.config = config
        self.device = device

        self.support_data = {}
        self.query_data = {}
        # Split 50/50 into support and query
        for k in data.keys():
            self.support_data[k], self.query_data[k] = np.array_split(data[k], 2)
            # Convert to Tensors
            self.support_data[k] = torch.tensor(
                self.support_data[k], dtype=torch.float32, device=self.device
            )
            self.query_data[k] = torch.tensor(
                self.query_data[k], dtype=torch.float32, device=self.device
            )

    def __len__(self):
        return len(self.support_data.keys())

    def __getitem__(self, idx):
        # Randomly flip query and support if they are not kept separate
        if not self.config["separate_query_and_support"] and random_sample() < 0.5:
            self.support_data, self.query_data = self.query_data, self.support_data

        # Pick class
        support_class = self.support_data[idx]
        query_class = self.query_data[idx]

        support_set = []
        query_set = []

        # Choose support_n random windows from class
        for i in randint(
            0,
            len(support_class) - self.config["window_width"],
            size=self.config["k_shot"],
        ):
            support_set.append(support_class[i : i + self.config["window_width"]])

        # Choose n_query random windows from class
        for i in randint(
            0, len(query_class) - self.config["window_width"], size=self.config["n_query"]
        ):
            query_set.append(query_class[i : i + self.config["window_width"]])

        support_set = torch.stack(support_set)
        query_set = torch.stack(query_set)
        support_query_set = torch.cat([support_set, query_set], dim=0)

        if self.config["FFT"]:
            support_query_set = torch.fft.rfft(support_query_set)
            support_query_set = torch.abs(support_query_set)

        if self.config["normalize"]:
            mean = (
                support_query_set.mean(dim=1)
                .reshape(-1, 1)
                .repeat(1, support_query_set.shape[1])
            )
            std = (
                support_query_set.std(dim=1)
                .reshape(-1, 1)
                .repeat(1, support_query_set.shape[1])
            )
            support_query_set = (support_query_set - mean) / std

        return support_query_set


def format_fewshot_classes(data, config):
    class_data = {}
    class_to_idx = {}
    class_counter = 0

    data = data.groupby(["rpm", "class"])

    for n, g in data:
        for sensor in g.columns:
            # Skip the non-sensors
            if sensor == "rpm" or sensor == "class":
                continue

            # Create class name
            # `rpm_class_sensor`
            class_name = "_".join((*[str(x) for x in n], sensor))

            # Save class name and number
            if not class_name in class_to_idx.keys():
                class_to_idx[class_name] = class_counter
                class_counter += 1

            # Save data
            #     Resulting class_to_idx:
            #     {
            #         "250_0_acc1": 0,
            #         ...
            #     }
            #     Resulting data format:
            #         0: [...],
            #     {
            #         ...
            #     }
            class_data[class_to_idx[class_name]] = g[sensor].to_numpy()

    return class_data, class_to_idx


def fewshot_data_selection(data, sensors, rpm, classes):
    selected_data = data.loc[:, sensors + ["rpm", "class"]]
    selected_data = selected_data[
        (selected_data["rpm"].isin(rpm)) & (selected_data["class"].isin(classes))
    ]

    return selected_data


class FewshotBatchSampler(Sampler):
    def __init__(self, config, class_num, rpm_num, sensor_num):
        self.config = config
        self.class_num = class_num
        self.rpm_num = rpm_num
        self.sensor_num = sensor_num

    def __iter__(self):
        class_perm = torch.arange(self.class_num)
        rpm_seq = torch.randint(high=self.rpm_num, size=(self.class_num,))
        sensor_seq = torch.randint(high=self.sensor_num, size=(self.class_num,))

        for i in range(math.floor(self.class_num / self.config["n_way"])):
            class_batch = class_perm[
                i * self.config["n_way"] : (i + 1) * self.config["n_way"]
            ]

            # RPM
            # "class" means each rpm forms a separate class,
            if self.config["rpm_status"] == "class":
                # "batch" means all final classes in a batch need to be with the same rpm
                rpm_batch = rpm_seq[i * self.config["n_way"] : (i + 1) * self.config["n_way"]]
            elif self.config["rpm_status"] == "batch":
                rpm_batch = [rpm_seq[i] for _ in range(self.config["n_way"])]
            # "any" case is resolved at dataset level, so these indices are discarded
            elif self.config["rpm_status"] == "any":
                rpm_batch = [0 for _ in range(self.config["n_way"])]

            # sensor
            if self.config["sensor_status"] == "class":
                sensor_batch = sensor_seq[
                    i * self.config["n_way"] : (i + 1) * self.config["n_way"]
                ]
            elif self.config["sensor_status"] == "batch":
                sensor_batch = [sensor_seq[i] for _ in range(self.config["n_way"])]
            elif self.config["sensor_status"] == "any":
                sensor_batch = [0 for _ in range(self.config["n_way"])]

            batch = list(zip(class_batch, rpm_batch, sensor_batch))
            yield batch

    # def __len__(self):
    #     return math.floor(self.class_num / self.config["n_way"])

def test_rig_timeseries_fewshot(config, device):
    # data_f = os.path.join(*config["data_path"], "arotor_old")
    print("Reading data")

    # Load data

    # data = pd.read_feather(f"{data_f}.feather")

    abs_path = os.path.dirname(__file__)
    data_folder = os.path.join(abs_path, os.pardir, os.pardir, "data")

    data = pd.read_feather(os.path.join(data_folder, "processed", "arotor_old.feather"))

    print("Preprocessing")

    # Create splits

    train_data = fewshot_data_selection(
        data, config["train_sensors"], config["train_rpm"], config["train_classes"]
    )
    validation_data = fewshot_data_selection(
        data,
        config["validation_sensors"],
        config["validation_rpm"],
        config["validation_classes"],
    )
    test_data = fewshot_data_selection(
        data, config["test_sensors"], config["test_rpm"], config["test_classes"]
    )

    # Format

    train_class_data, class_to_idx = format_fewshot_classes(train_data, config)
    validation_class_data, _ = format_fewshot_classes(validation_data, config)
    test_class_data, _ = format_fewshot_classes(test_data, config)

    print()
    print("class_to_idx mapping:")
    print(class_to_idx)

    # Create datasets

    train_dataset = FewShotDataset(train_class_data, config, device)
    validation_dataset = FewShotDataset(validation_class_data, config, device)
    test_dataset = FewShotDataset(test_class_data, config, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["n_way"],
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["n_way"],
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["n_way"],
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    return train_loader, validation_loader, test_loader


def get_arotor_old_data(config, device):
    print("Starting data loading")

    train_loader = None
    validation_loader = None
    test_loader = None

    train_loader, validation_loader, test_loader = test_rig_timeseries_fewshot(config, device)

    print("Data processing done")

    return train_loader, validation_loader, test_loader
