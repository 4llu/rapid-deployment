import math
import os
from numpy.random import random_sample, randint

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


# class FewShotDataset(Dataset):
#     def __init__(self, data, config, device):
#         self.config = config
#         self.device = device

#         self.support_data = {}
#         self.query_data = {}
#         # Split 50/50 into support and query
#         for k in data.keys():
#             self.support_data[k], self.query_data[k] = np.array_split(data[k], 2)
#             # Convert to Tensors
#             self.support_data[k] = torch.tensor(
#                 self.support_data[k], dtype=torch.float32, device=self.device
#             )
#             self.query_data[k] = torch.tensor(
#                 self.query_data[k], dtype=torch.float32, device=self.device
#             )

#     def __len__(self):
#         return len(self.support_data.keys())

#     def __getitem__(self, idx):
#         # Randomly flip query and support if they are not kept separate
#         if not self.config["separate_query_and_support"] and random_sample() < 0.5:
#             self.support_data, self.query_data = self.query_data, self.support_data

#         # Pick class
#         support_class = self.support_data[idx]
#         query_class = self.query_data[idx]

#         support_set = []
#         query_set = []

#         # Choose support_n random windows from class
#         for i in randint(
#             0,
#             len(support_class) - self.config["window_width"],
#             size=self.config["k_shot"],
#         ):
#             support_set.append(support_class[i : i + self.config["window_width"]])

#         # Choose n_query random windows from class
#         for i in randint(
#             0,
#             len(query_class) - self.config["window_width"],
#             size=self.config["n_query"],
#         ):
#             query_set.append(query_class[i : i + self.config["window_width"]])

#         support_set = torch.stack(support_set)
#         query_set = torch.stack(query_set)
#         support_query_set = torch.cat([support_set, query_set], dim=0)

#         if self.config["FFT"]:
#             support_query_set = torch.fft.rfft(support_query_set)
#             support_query_set = torch.abs(support_query_set)

#         if self.config["normalize"]:
#             mean = (
#                 support_query_set.mean(dim=1)
#                 .reshape(-1, 1)
#                 .repeat(1, support_query_set.shape[1])
#             )
#             std = (
#                 support_query_set.std(dim=1)
#                 .reshape(-1, 1)
#                 .repeat(1, support_query_set.shape[1])
#             )
#             support_query_set = (support_query_set - mean) / std

#         return support_query_set


class FewShotDatasetMixed(Dataset):
    def __init__(self, support, query, config, device, split):
        self.support = support
        self.query = query
        self.config = config
        self.device = device
        self.split = split

        self.class_num = len(self.config[split + "_classes"])
        self.rpm_num = len(self.config[split + "_rpm"])
        self.sensor_num = len(self.config[split + "_sensors"])

        self.max_measurement_index = math.ceil(
            (self.support.shape[-1] - self.config["window_width"])
            / ((1 - self.config["window_overlap"]) * self.config["window_width"])
        )
        # Length

        # Base length (just the original classes)
        self.length = self.support.shape[0]
        # RPMs separate
        # if self.config["tatus"] == "class" or self.config["rpm_status"] == "batch":
        self.length *= self.support.shape[1]
        # Sensors separate
        # if (
        #     self.config["sensor_status"] == "class"
        #     or self.config["sensor_status"] == "batch"
        # ):
        self.length *= self.support.shape[2]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # idx -> (class, rpm, sensor)
        # Define samples
        sample_classes = (
            torch.ones(self.config["k_shot"] + self.config["n_query"], dtype=torch.long)
            * idx[0]
        )

        if self.config["rpm_status"] == "any":
            sample_rpms = torch.randint(
                high=self.rpm_num,
                size=(self.config["k_shot"] + self.config["n_query"],),
                dtype=torch.long,
            )
        else:
            sample_rpms = (
                torch.ones(
                    self.config["k_shot"] + self.config["n_query"], dtype=torch.long
                )
                * idx[1]
            )

        if self.config["sensor_status"] == "any":
            sample_sensors = torch.randint(
                high=self.sensor_num,
                size=(self.config["k_shot"] + self.config["n_query"],),
                dtype=torch.long,
            )
        else:
            sample_sensors = (
                torch.ones(
                    self.config["k_shot"] + self.config["n_query"], dtype=torch.long
                )
                * idx[2]
            )

        sample_idxs = torch.randperm(self.max_measurement_index, dtype=torch.long)[
            : self.config["k_shot"] + self.config["n_query"]
        ]

        # Get samples

        support = []
        query = []
        sample_definitions = list(
            zip(sample_classes, sample_rpms, sample_sensors, sample_idxs)
        )
        for c, r, s, i in sample_definitions[: self.config["k_shot"]]:
            support.append(self.support[c, r, s, i : i + self.config["window_width"]])
        for c, r, s, i in sample_definitions[self.config["k_shot"] :]:
            query.append(self.query[c, r, s, i : i + self.config["window_width"]])

        support_query_set = torch.stack(support + query, dim=0)

        # Possible preprocessing

        if self.config["FFT"]:
            print(support_query_set.shape)
            print(support_query_set.dtype)
            print(support_query_set.device)
            support_query_set = torch.fft.rfft(support_query_set)
            support_query_set = torch.abs(support_query_set)
            quit()

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

        # Swap support and query sets to mix samples between batches
        if not self.config["separate_query_and_support"]:
            self.support, self.query = self.query, self.support

        return support_query_set


# def format_fewshot_classes(data, config):
#     class_data = {}
#     class_to_idx = {}
#     class_counter = 0

#     data = data.groupby(["rpm", "class"])

#     for n, g in data:
#         for sensor in g.columns:
#             # Skip the non-sensors
#             if sensor == "rpm" or sensor == "class":
#                 continue

#             # Create class name
#             # `rpm_class_sensor`
#             class_name = "_".join((*[str(x) for x in n], sensor))

#             # Save class name and number
#             if not class_name in class_to_idx.keys():
#                 class_to_idx[class_name] = class_counter
#                 class_counter += 1

#             # Save data
#             #     Resulting class_to_idx:
#             #     {
#             #         "250_0_acc1": 0,
#             #         ...
#             #     }
#             #     Resulting data format:
#             #         0: [...],
#             #     {
#             #         ...
#             #     }
#             class_data[class_to_idx[class_name]] = g[sensor].to_numpy()

#     return class_data, class_to_idx


def format_fewshot_classes_mixed(data, config, device):
    formatted_data = []
    class_map = []
    class_i = 0
    prev_class = ""
    rpm_map = []
    rpm_i = 0
    all_rpms_added = False
    sensor_map = []
    sensor_i = 0
    all_sensors_added = False
    min_sensor_len = 9999999999

    data = data.groupby(["class", "rpm"])

    for n, g in data:
        # First case
        if prev_class == "":
            prev_class = n[0]
            # Start things off
            class_map.append((class_i, n[0]))
            formatted_data.append([])
        # Next class
        elif prev_class != n[0]:
            class_i += 1
            # New class stuff
            class_map.append((class_i, n[0]))
            formatted_data.append([])
            # Restart rpm counting
            rpm_i = 0
            # No need to keep track of these after first class has been gone through
            all_rpms_added = True

        formatted_data[class_i].append([])

        # Add all sensors
        for sensor in g.columns:
            # Non-sensor columns
            if sensor == "rpm" or sensor == "class":
                continue

            # Keep track of shortest measurement
            if len(g[sensor]) < min_sensor_len:
                min_sensor_len = len(g[sensor])

            formatted_data[class_i][rpm_i].append(list(g[sensor]))

            if not all_sensors_added:
                sensor_map.append([sensor_i, sensor])
                sensor_i += 1

        # Keep track of rpm mappings
        if not all_rpms_added:
            rpm_map.append([rpm_i, n[1]])
        rpm_i += 1

        # No need to keep track of these after first rpm has been gone through
        all_sensors_added = True

        # Keep track of class changes as they aren't in their own loop
        prev_class = n[0]

    # Make sure everything is of the same length

    if min_sensor_len % 2 != 0:
        # Make the usable measurement len even to make splitting easier
        min_sensor_len -= 1
    for i in range(len(formatted_data)):
        for j in range(len(formatted_data[i])):
            for k in range(len(formatted_data[i][j])):
                formatted_data[i][j][k] = formatted_data[i][j][k][:min_sensor_len]

    # Convert to tensor
    formatted_data = torch.tensor(formatted_data, device=device)
    formatted_support, formatted_support = torch.tensor_split(formatted_data, 2, dim=3)

    # Print info
    print("Data size:", formatted_data.shape)
    print("Class_map:", class_map)
    print("RPM map:", rpm_map)
    print("Sensor map:", sensor_map)

    return formatted_support, formatted_support


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
                rpm_batch = rpm_seq[
                    i * self.config["n_way"] : (i + 1) * self.config["n_way"]
                ]
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

    print()
    print("Train data")
    print()
    train_support, train_query = format_fewshot_classes_mixed(
        train_data, config, device
    )
    print()
    print("Validation data")
    print()
    validation_support, validation_query = format_fewshot_classes_mixed(
        validation_data, config, device
    )
    print()
    print("Test data")
    print()
    test_support, test_query = format_fewshot_classes_mixed(test_data, config, device)

    # Create datasets

    train_dataset = FewShotDatasetMixed(
        train_support, train_query, config, device, "train"
    )
    validation_dataset = FewShotDatasetMixed(
        validation_support, validation_query, config, device, "validation"
    )
    test_dataset = FewShotDatasetMixed(test_support, test_query, config, device, "test")

    # Create samplers

    train_sampler = FewshotBatchSampler(
        config,
        len(config["train_classes"]),
        len(config["train_rpm"]),
        len(config["train_sensors"]),
    )
    validation_sampler = FewshotBatchSampler(
        config,
        len(config["validation_classes"]),
        len(config["validation_rpm"]),
        len(config["validation_sensors"]),
    )
    test_sampler = FewshotBatchSampler(
        config,
        len(config["test_classes"]),
        len(config["test_rpm"]),
        len(config["test_sensors"]),
    )

    # Create dataloaders

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        pin_memory=False,
    )

    return train_loader, validation_loader, test_loader


def get_arotor_old_data(config, device):
    print("Starting data loading")

    train_loader = None
    validation_loader = None
    test_loader = None

    train_loader, validation_loader, test_loader = test_rig_timeseries_fewshot(
        config, device
    )

    print("Data processing done")

    return train_loader, validation_loader, test_loader
