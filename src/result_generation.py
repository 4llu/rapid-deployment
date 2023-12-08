import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from main import run_training, setup_data, setup_device
from models.models import setup_model
from training.trainers import target_converter
from utils.config import setup_config


def main():
    # INIT
    ######

    device = setup_device()

    # Init arguments
    parser = ArgumentParser()
    parser.add_argument(f"--config", type=type("a"), required=True)
    parser.add_argument(f"--config_override_base", type=type("a"), required=True)
    parser.add_argument(f"--array_task_id", default=1, type=type(1))
    parser.add_argument(f"--block_size", default=1, type=type(1))
    parser.add_argument(f"--repetitions", default=1, type=type(1))
    parser.add_argument(f"--ensemble_size", default=1, type=type(1))
    parser.add_argument(f"--job_name", type=type("a"))

    # Parse args
    args = parser.parse_args()

    # Determine result directory
    folder_path = None
    if args.job_name is None:
        time = datetime.now().strftime("%m-%d_%H-%M-%S")
        folder_path = os.path.join("reports", "RAW", "result_generation", time)
    else:
        folder_path = os.path.join("reports", "RAW", "result_generation", args.job_name)

    # Setup result directory (if it doesn't exist)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # CONFIG RUNS
    ##############

    for j in range(args.block_size):
        run_config_id = (args.array_task_id - 1) * args.block_size + j

        # Check that override config exists beforehand, because block size might not match with total
        # number of configurations to run)
        override_filename = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "configs",
            *(f"{args.config_override_base}_{run_config_id}.yaml").split("/"),
        )
        if not os.path.isfile(override_filename):
            print(override_filename)
            print("######################################")
            print("All configurations complete, quitting!")
            print("######################################")
            quit()

        # Read config
        config = setup_config(
            args.config,
            config_override_name=f"{args.config_override_base}_{run_config_id}",
        )

        # Initialize data
        train_loader, validation_loader, test_loader = setup_data(config, device)

        accuracies = []  # * Not actually necessary, but convenient
        cfs = []
        ensemble_cfs = []
        first_cfs = []

        # REPETITION RUNS
        #################

        for i in range(args.repetitions):
            model_weights = []

            for k in range(args.ensemble_size):
                print()
                print("############################")
                print(
                    f"ENSEMBLE MODEL {k + 1} OF REPETITION {i + 1} OF CONFIG ID {run_config_id}"
                )
                print("############################")
                print()
                _, test_accuracy, test_cf, weights = run_training(
                    train_loader,
                    validation_loader,
                    test_loader,
                    config,
                    device=device,
                    run_type="result",
                )
                if k == 0:
                    accuracies.append(test_accuracy)
                    cfs.append(test_cf)
                model_weights.append(weights)

            # ENSEMBLE
            ##########

            if config["backbone"] != "InceptionTime":
                raise Exception("Model selection not implemented yet!")
            models = []
            for weights in model_weights:
                model = setup_model(config, device)
                model = model.to(device)
                model.load_state_dict(weights)
                model.eval()

                models.append(model)

            first_cfs_repetitions = []
            ensemble_cfs_repetitions = []
            for n in range(config["test_episodes"]):
                test_features, test_labels = next(iter(test_loader))
                test_features = test_features.to(device, non_blocking=True)

                test_labels = list(zip(*test_labels))[0]
                y = target_converter(test_labels, config, device)

                y_preds = []
                for model in models:
                    y_pred, _, _, _ = model(test_features)

                    y_preds.append(F.softmax(y_pred, dim=-1))

                # First of ensemble (for checking)
                y_pred_first = torch.argmax(y_preds[0], dim=-1)
                cf_first = confusion_matrix(
                    y.cpu().detach().numpy(), y_pred_first.cpu().detach().numpy()
                )
                first_cfs_repetitions.append(cf_first)

                # Ensemble results
                y_preds = torch.stack(y_preds, dim=1)
                y_preds = y_preds.mean(dim=1)
                y_preds = torch.argmax(y_preds, dim=-1)

                cf = confusion_matrix(
                    y.cpu().detach().numpy(), y_preds.cpu().detach().numpy()
                )
                ensemble_cfs_repetitions.append(cf)

            # Ensemble first
            first_cfs_repetitions = np.stack(first_cfs_repetitions, axis=0)
            first_cfs_repetitions = first_cfs_repetitions.sum(axis=0)
            first_cfs.append(first_cfs_repetitions)
            # Ensemble all
            ensemble_cfs_repetitions = np.stack(ensemble_cfs_repetitions, axis=0)
            ensemble_cfs_repetitions = ensemble_cfs_repetitions.sum(axis=0)
            ensemble_cfs.append(ensemble_cfs_repetitions)

        # SAVE RESULTS
        ##############

        joblib.dump(
            {
                "accuracies": accuracies,
                "cfs": cfs,
                "ensemble_cfs": ensemble_cfs,
                "first_cfs": first_cfs,
            },
            os.path.join(folder_path, f"results_for_config_{run_config_id}.pkl"),
        )


if __name__ == "__main__":
    main()
