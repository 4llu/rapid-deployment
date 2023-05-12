import os
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat

import joblib
import numpy as np
import optuna
from optuna.trial import TrialState

from main import setup_data, setup_device, run_training
from utils.config import setup_config


def main():
    # INIT
    #######

    device = setup_device()

    # Init arguments
    parser = ArgumentParser()
    parser.add_argument(f"--config", default="arotor/h_optimize", type=type("a"))
    parser.add_argument(f"--n_trials", default=3, type=type(2))
    parser.add_argument(f"--n_repetitions", default=1, type=type(2))

    # Parse args
    args = parser.parse_args()
    n_trials = args.n_trials
    n_repetitions = args.n_repetitions

    # Read config
    config = setup_config(args.config)

    # Initialize data
    train_loader, validation_loader, test_loader = setup_data(config, device)

    # SETUP STUDY
    #############

    study_name = (
        f"{config['name']}_{config['data']}_{datetime.now().strftime('%m-%d_%H-%M-%S')}"
    )
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        study_name=study_name, direction="maximize", pruner=pruner
    )

    # RUN
    #####

    # Run through all trials
    for i in range(n_trials):
        trial = study.ask()

        # TEST HYPERPARAMETERS
        ######################

        config["lr"] = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        config["momentum"] = trial.suggest_float(
            "momentum",
            0.9,
            0.99,
        )
        config["weight_decay"] = trial.suggest_float(
            "weight_decay", 0.00001, 0.001, log=True
        )
        config["sch_gamma"] = trial.suggest_float("sch_gamma", 0.95, 0.9998, log=True)
        config["cl_dropout"] = trial.suggest_float("cl_dropout", 0.0, 0.6, step=0.1)
        config["fc_dropout"] = trial.suggest_float("fc_dropout", 0.0, 0.6, step=0.1)
        config["embedding_multiplier"] = trial.suggest_categorical(
            "embedding_multiplier", [1, 10, 100, 1000, 10000]
        )

        # Repeat training with same set of hyperparameters for validity
        # NOTE This doesn't really work with pruning
        accuracies = []
        for j in range(n_repetitions):
            print()
            print("############################")
            print(f"REPETITION {j} OF TRIAL {i}")
            print("############################")
            print()
            val_accuracy = run_training(
                train_loader,
                validation_loader,
                test_loader,
                config,
                device=device,
                trial=True,
            )
            accuracies.append(val_accuracy)

        # Report trial results
        accuracy = np.array(accuracies).mean()
        study.tell(trial, accuracy)

    # CONCLUSION
    ############

    # Save study
    joblib.dump(
        study, os.path.join("reports", "raw", "optuna_studies", f"{study_name}.pkl")
    )

    # Print results

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print()
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    importances = optuna.importance.get_param_importances(study)
    print()
    print("Hyperparameter importances: \n", pformat(importances))

    print()
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
