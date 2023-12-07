import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import joblib

from main import run_training, setup_data, setup_device
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

        # REPETITION RUNS
        #################

        for i in range(args.repetitions):
            print()
            print("############################")
            print(f"REPETITION {i + 1} OF CONFIG ID {run_config_id}")
            print("############################")
            print()
            _, test_accuracy, test_cf = run_training(
                train_loader,
                validation_loader,
                test_loader,
                config,
                device=device,
                run_type="trial",
            )
            accuracies.append(test_accuracy)
            cfs.append(test_cf)

        # SAVE RESULTS
        ##############

        joblib.dump(
            {"accuracies": accuracies, "cfs": cfs},
            os.path.join(folder_path, f"results_for_config_{run_config_id}.pkl"),
        )


if __name__ == "__main__":
    main()
