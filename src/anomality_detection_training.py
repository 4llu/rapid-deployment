from argparse import ArgumentParser

from main import run_training, setup_data, setup_device
from utils.config import setup_config


def main():
    # INIT
    ######

    device = setup_device()

    # Init arguments
    parser = ArgumentParser()
    parser.add_argument(f"--config", type=type("a"), required=True)
    # parser.add_argument(f"--array_task_id", default=1, type=type(1))
    parser.add_argument(f"--repetitions", default=1, type=type(1))
    parser.add_argument(f"--job_id", type=type("a"))

    # Parse args
    args = parser.parse_args()

    # CONFIG RUNS
    ##############

    # Read config
    config = setup_config(args.config)
    config.update({"job_id": args.job_id})

    # Initialize data
    train_loader, validation_loader, test_loader = setup_data(config, device)

    # REPETITION RUNS
    #################

    for i in range(args.repetitions):
        print()
        print("############################")
        print(f"REPETITION {i + 1}")
        print("############################")
        print()
        _, _, _ = run_training(
            train_loader,
            validation_loader,
            test_loader,
            {**config, **{"i": i}},
            device=device,
            run_type="trial",
        )


if __name__ == "__main__":
    main()
