import os
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat

import joblib
import numpy as np
import optuna
from optuna.trial import TrialState

from main import run
from utils.config import setup_config


def objective(trial, config, n_repetitions, run_training_partial):
    # Traditional hyperparameters
    #############################

    config["lr"] = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)

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

    # Preprocessing & Augmentation
    ##############################

    # FIXME
    # trial.suggest_categorical("pre_aug", [["rolling_average"], ["low_pass_filter"], []])

    # config["preprocessing"] = trial.suggest_categorical(
    #     "preprocessing", [["rolling_average"], ["low_pass_filter"], []]
    # )

    # if "rolling_average" in config["preprocessing"]:
    #     config["rolling_average_window"] = trial.suggest_int(
    #         "rolling_average_window", 2, 50, 5
    #     )

    # if "low_pass_filter" in config["preprocessing"]:
    #     config["low_pass_filter_cutoff"] = trial.suggest_int(
    #         "low_pass_filter_cutoff", 100, 1000, 100
    #     )

    # Signals
    #########

    # FIXME Doesn't work with the way data is handled currently
    # config["signals"] = trial.suggest_categorical(
    #     "signals", [["angle", "speed"], ["angle", "speed", "acc"], ["angle", "speed", "acc_time"]]
    # )

    # Run multiple times with same parameters to take stability into account
    accuracies = []
    for i in range(n_repetitions):
        # Run
        accuracies.append(run_training_partial(config))

    accuracy = np.array(accuracies).mean()

    # Return results for parent process
    return accuracy


def main():
    # INIT
    #######

    # Init arguments
    parser = ArgumentParser()
    parser.add_argument(f"--config", default="sim2real/h_optimize", type=type("a"))
    parser.add_argument(f"--n_trials", default=3, type=type(2))
    parser.add_argument(f"--n_repetitions", default=1, type=type(2))

    # Parse args
    args = parser.parse_args()

    n_trials = args.n_trials
    n_repetitions = args.n_repetitions
    
    # Read config
    config = setup_config(args.config)

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

    run(config, {"study": study, "objective": objective, "n_trials": n_trials, "n_repetitions": n_repetitions})

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
