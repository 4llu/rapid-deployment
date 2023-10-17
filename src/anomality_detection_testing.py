import os
import platform
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from data.arotor import fewshot_data_selection
from data.arotor_replication import data_selection_baseline, data_selection_faults
from main import setup_device
from models.backbones.inception_time import InceptionTime
from utils.config import setup_config


def get_AD_arotor_data(config, data_folder, device):
    # Load all data
    data = pd.read_feather(os.path.join(data_folder, "processed", "arotor.feather"))

    # Select support data (baseline measurements only)
    ##
    support_data = fewshot_data_selection(
        data, config["support_sensors"], config["support_rpm"], config["support_classes"]
    )

    support_data = (
        support_data.groupby(by=["rpm", "class"], observed=True)[config["support_sensors"][0]].apply(list).to_dict()
    )
    # Convert to tensors
    for k, v in support_data.items():
        # * Only use the first half of baseline data
        support_data[k] = torch.tensor(v, device=device)[: len(v) // 2]

    # Select query data
    ##
    query_data = fewshot_data_selection(data, config["query_sensors"], config["query_rpm"], config["query_classes"])

    query_data = (
        query_data.groupby(by=["rpm", "class"], observed=True)[config["query_sensors"][0]].apply(list).to_dict()
    )
    # Convert to tensors
    for k, v in query_data.items():
        new_tensor = torch.tensor(v, device=device)
        # Use second half of baseline measurements
        if k[1] == 0:
            new_tensor = new_tensor[len(v) // 2 :]
        query_data[k] = new_tensor

    return support_data, query_data


def get_AD_arotor_replication_data(config, data_folder, device):
    # Support data (baseline measurements only)

    support_data = data_selection_baseline(
        data_folder,
        config["support_sensors"],
        config["support_baseline_GPs"],
        config["support_rpms"],
        config["support_torques"],
    )

    # Group
    support_data = (
        support_data.groupby(by=["severity", "rpm", "torque"], observed=True)[config["support_sensors"][0]]
        .apply(list)
        .to_dict()
    )
    # Convert to tensors
    for k, v in support_data.items():
        support_data[k] = torch.tensor(v, device=device)

    # Query data (faults and remaining baselines)

    query_fault_data = data_selection_faults(
        data_folder,
        config["query_sensors"],
        config["query_faults"],
        config["query_severities"],
        config["query_rpms"],
        config["query_torques"],
        config["query_installations"],
    )

    query_baseline_data = data_selection_baseline(
        data_folder,
        config["query_sensors"],
        config["query_baseline_GPs"],
        config["query_rpms"],
        config["query_torques"],
    )

    # Join
    query_data = pd.concat([query_fault_data, query_baseline_data])

    # Group
    query_data = (
        query_data.groupby(by=["fault", "severity", "rpm", "torque", "installation"], observed=True)[
            config["support_sensors"][0]
        ]
        .apply(list)
        .to_dict()
    )
    # Convert to tensors
    for k, v in query_data.items():
        query_data[k] = torch.tensor(v, device=device, dtype=torch.float32)

    return support_data, query_data


def setup_AD_data(config, data_folder, device):
    if config["data"] == "ARotor":
        support_data, query_data = get_AD_arotor_data(config, data_folder, device)
    elif config["data"] == "ARotor_replication":
        support_data, query_data = get_AD_arotor_replication_data(config, data_folder, device)
    else:
        raise Exception("No such data configuration as`", config["data"], "`!")

    return support_data, query_data


def main():

    # INITIALIZATION
    ################

    # Init arguments
    parser = ArgumentParser()
    parser.add_argument(f"--config", default="anomality_detection/arotor_replication_test", type=type("a"))

    # Parse args
    args = parser.parse_args()

    # Read config
    config = setup_config(args.config)

    # Determine device
    device = setup_device()

    # DATA
    ######
    print("Loading data")

    # Initialize data
    abs_path = os.path.dirname(__file__)
    data_folder = os.path.join(abs_path, os.pardir, "data")

    support_data, query_data = setup_AD_data(config, data_folder, device)

    # TODO Preprocessing (at least `full`)

    # MODEL
    #######
    print("Preparing model")

    # TODO Further model selection
    if config["backbone"] != "InceptionTime":
        raise Exception("Model selection not implemented yet!")

    model = InceptionTime(config)
    model = model.to(device)

    # model = setup_model(config, device)

    # * torch.compile doesn't work on windows currently
    if device.type == "cuda" and platform.system() != "Windows":
        print()
        print("#####################")
        print("Using torch.compile()")
        print("#####################")
        print()
        model = torch.compile(model)

    # EMBED
    #######
    print("Beginning embedding")

    save_folder = Path(
        os.path.join(
            abs_path,
            os.pardir,
            "reports",
            "RAW",
            "embedding_databases",
            f"{config['name']}_{datetime.now().strftime('%m-%d_%H-%M-%S')}",
        )
    )
    save_folder.mkdir(parents=True, exist_ok=True)

    i = 0
    for root, _, files in os.walk(os.path.join(abs_path, os.pardir, "model_weights", config["model_weight_dir"])):
        # Make sure the best weights are last (sorted by ascending accuracy)
        files.sort()

        # TODO History ensemble support
        if not "model_" in files[-1]:
            continue

        # Load weights
        model_weights = torch.load(
            os.path.join(os.pardir, root, files[-1]),
            map_location=torch.device(device),
        )

        new_model_weights = {}
        for k in model_weights.keys():
            new_model_weights[k.replace("_orig_mod.backbone.", "")] = model_weights[k]
        model.load_state_dict(new_model_weights)
        model.eval()

        # Process support
        print("Embedding supports")
        support_vector_df = []

        # TODO (Maybe) No overlap option currently
        for k, v in support_data.items():
            # Window
            samples = v[: -(len(v) % config["window_width"])].view(-1, 1, config["window_width"])
            # Embed
            embeddings = model(samples)
            # Remove useless dimensions
            embeddings = embeddings.squeeze()
            # Multiplier
            embeddings = embeddings * config["embedding_multiplier"]
            # Save
            df = pd.DataFrame(embeddings.cpu().detach())
            if config["data"] == "ARotor":
                df["rpm"] = k[0]
                df["class"] = k[1]
            elif config["data"] == "ARotor_replication":
                df["GP"] = k[0]
                df["rpm"] = k[1]
                df["torque"] = k[2]
            else:
                raise Exception("WAT?")
            support_vector_df.append(df)

        support_vector_df = pd.concat(support_vector_df)

        # TODO Save
        support_vector_df.to_feather(
            os.path.join(
                save_folder,
                f"support_{i}.feather",
            )
        )

        # Process queries
        print("Embedding queries")
        query_vector_df = []

        # TODO (Maybe) No overlap option currently
        for k, v in query_data.items():
            print(k)
            # Window
            extra_at_end = len(v) % config["window_width"]
            # :-0 edge case
            if extra_at_end != 0:
                samples = v[:-extra_at_end]
            else:
                samples = v
            samples = samples.view(-1, 1, config["window_width"])

            # Embed
            with torch.no_grad():
                embeddings = model(samples)
            # Remove useless dimensions
            embeddings = embeddings.squeeze()
            # Multiplier # * Probably has no effect
            embeddings = embeddings * config["embedding_multiplier"]
            # Save
            df = pd.DataFrame(embeddings.cpu())
            if config["data"] == "ARotor":
                df["rpm"] = k[0]
                df["class"] = k[1]
            elif config["data"] == "ARotor_replication":
                df["fault"] = k[0]
                df["severity"] = k[1]
                df["rpm"] = k[2]
                df["torque"] = k[3]
                df["installation"] = k[4]
            else:
                raise Exception("WAT?")
            query_vector_df.append(df)

        query_vector_df = pd.concat(query_vector_df)

        # print(support_vector_df.shape)
        # print(support_vector_df.head(3))
        # print(query_vector_df.shape)
        # print(query_vector_df.head(3))

        if config["data"] == "ARotor_replication":
            query_vector_df["fault"] = query_vector_df["fault"].astype("str")
            query_vector_df["severity"] = query_vector_df["severity"].astype("str")

        # TODO Save
        query_vector_df.to_feather(
            os.path.join(
                save_folder,
                f"query_{i}.feather",
            )
        )
        i += 1


if __name__ == "__main__":
    main()
