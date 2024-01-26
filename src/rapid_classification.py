import os
from argparse import ArgumentParser

import joblib
import numpy as np
import torch
import tqdm
from sklearn.metrics import confusion_matrix

from data.data import setup_data
from main import setup_device
from models.models import setup_model
from training.trainers import target_converter
from utils.config import setup_config


def run_testing(
    config,
):
    # INITIALIZATION
    ################

    device = setup_device()

    # DATA
    ######

    # Initialize data
    #! FIXME
    _, _, test_loader = setup_data(config, device)

    # MODEL INITIALIZATION
    ######################

    model = setup_model(config, device)
    # ? torch.compile() intentionally left out, because of the small number of epochs to be run

    # Get model weights
    ##

    abs_path = os.path.dirname(__file__)
    all_ensemble_weights = []
    i = 0
    for root, _, files in os.walk(
        os.path.join(abs_path, os.pardir, "model_weights", config["model_weight_dir"])
    ):
        # Make sure the best weights are last in case multiple checkpoints saved (sorted by ascending accuracy)
        files.sort()
        # Make sure this is a model weight file
        if len(files) < 1 or not "model_" in files[-1]:
            continue

        # Start next ensemble
        if i % config["ensemble_size"] == 0:
            all_ensemble_weights.append([])
        i += 1

        # Load weights
        model_weights = torch.load(
            os.path.join(os.pardir, root, files[-1]),
            map_location=torch.device(device),
        )
        new_model_weights = {}
        for k in model_weights.keys():
            new_model_weights[k.replace("_orig_mod.", "")] = model_weights[k]

        all_ensemble_weights[-1].append(new_model_weights)

    # TEST
    ######

    all_accuracies = []
    all_cfs = []

    # Go through all ensembles
    ensemble_num = 1
    for ensemble_weights in all_ensemble_weights:
        print(f"ENSEMBLE #{ensemble_num}")
        # * The below hack because we need the same batches for each ensemble model, but switcing model
        # * weights every episode is slow and saving all models separately on the GPU uses a lot of
        # * GPU memory
        # Get all episodes
        episodes = []
        for i in range(config["test_episodes"]):
            # * This is kind of a stupid loop as the dataloader only has length for one batch per loop
            for batch in test_loader:
                # batch = next(test_loader)
                samples = batch[0].to(device, non_blocking=True)
                # Conversion because the labels are returned as a (class, rpm, sensor) tuple
                # and we only care about the class here
                targets = list(zip(*batch[1]))[0]
                # Original targets are episode classes, not including that there are n_query queries
                # per class
                targets = target_converter(targets, config, device)

                episodes.append((samples, targets))

        # Gather targets as they are the same for all
        ensemble_targets = []
        for _, targets in episodes:
            ensemble_targets.append(targets)
        ensemble_targets = torch.stack(ensemble_targets, dim=0)
        # print(ensemble_targets.shape)

        # Compute predictions for all ensembles
        ensemble_predictions = []
        for model_weights in tqdm.tqdm(
            ensemble_weights, position=0, desc="Ensemble weights"
        ):
            ensemble_predictions.append([])

            model.load_state_dict(model_weights)
            model.eval()

            # Go through all episodes
            for samples, _ in tqdm.tqdm(episodes, position=1, desc="Episodes"):
                with torch.no_grad():
                    predictions, _, _, _ = model(samples)
                ensemble_predictions[-1].append(predictions)

            ensemble_predictions[-1] = torch.stack(ensemble_predictions[-1], dim=0)

        ensemble_predictions = torch.stack(ensemble_predictions, dim=1)
        ensemble_predictions = torch.sum(ensemble_predictions, dim=1)
        _, ensemble_predictions = torch.max(ensemble_predictions, dim=-1)

        # Save accuracies
        accuracies = (ensemble_predictions == ensemble_targets).sum(dim=-1) / (
            config["n_way"] * config["n_query"]
        )
        all_accuracies.append(accuracies)
        # Save cfs
        cfs = []
        for ep in range(ensemble_predictions.shape[0]):
            cf = confusion_matrix(
                ensemble_predictions[ep].cpu().detach().numpy(),
                ensemble_targets[ep].cpu().detach().numpy(),
            )
            cfs.append(cf)

        cfs = np.stack(cfs, axis=0)
        all_cfs.append(cfs)

        ensemble_num += 1

    all_accuracies = torch.cat(all_accuracies)
    all_accuracies = all_accuracies.cpu().numpy()
    print(all_accuracies)
    print(all_accuracies.mean())

    all_cfs = np.concatenate(all_cfs)

    return all_accuracies, all_cfs


if __name__ == "__main__":
    # Init arguments
    parser = ArgumentParser()
    parser.add_argument(
        f"--config",
        default="rapid_classification/arotor_replication_test",
        type=type("a"),
    )

    # Parse args
    args = parser.parse_args()

    # Read config
    config = setup_config(args.config)

    # Run setup
    all_accuracies, all_cfs = run_testing(config)

    joblib.dump(
        {
            "accuracies": all_accuracies,
            "cfs": all_cfs,
        },
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "reports",
            "RAW",
            "rapid_classification_results",
            f"{config['name']}_{config['data']}_{config['test_sensors'][0]}.pkl",
        ),
    )
