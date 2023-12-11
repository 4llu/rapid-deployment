import os
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat

import torch
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from ignite.metrics.metric import MetricUsage

from data.data import setup_data
from models.models import setup_model
from training.trainers import setup_evaluator, setup_trainer
from utils.config import setup_config
from utils.custom_metrics import Confusion_matrices, RNFSAccuracy
from utils.logging import log_metrics, setup_logging


def run_training(
    train_loader,
    validation_loader,
    test_loader,
    config,
    device=None,
    run_type="normal",
):
    # * Use this to run a single training
    # FIXME HERE PROBLEM XXX Could this be moved to higher level?
    # Requires finding out how to reinitialize model weights
    # INITIALIZE MODEL
    ##################

    model = setup_model(config, device)

    # TRAINING PREPARATION
    ######################

    # Optimizers

    # Adam and AdamW
    if "Adam" in config["optimizer"]:
        optimizer = getattr(torch.optim, config["optimizer"])(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            betas=[config["momentum"], 0.999],  # 0.999 is the default beta2
        )
    # SGD
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
        )

    # Scheduler
    scheduler = None

    if config["lr_scheduler"] == "exponential_lr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            config["sch_gamma"],
            verbose=False,
        )

    elif config["lr_scheduler"] == "one_cycle_lr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=config["max_epochs"],
            steps_per_epoch=20,  # FIXME
            max_lr=config["max_lr"],
            div_factor=config["max_lr"] / config["min_lr"],
            final_div_factor=config["final_div_factor"],
            three_phase=config["three_phase"],
            pct_start=config["pct_start"],
            anneal_strategy=config["anneal_strategy"],
            cycle_momentum=config["cycle_momentum"],
            base_momentum=config["base_momentum"],
            max_momentum=config["max_momentum"],
            verbose=False,
        )

    # Loss function
    # ? Moved to trainer
    if config["model"] == "prototypical":
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    elif config["model"] == "relation":
        loss_fn = torch.nn.MSELoss()
    else:
        raise Exception("WAT")

    loss_fn = loss_fn.to(device=device)

    # TRAINING
    ##########

    # Setup trainers
    trainer = setup_trainer(config, model, optimizer, loss_fn, device)  # Training
    evaluator = setup_evaluator(config, model, device, "validation")  # Validation
    evaluator_test = setup_evaluator(config, model, device, "test")  # Testing

    # Enable the use of multiple episodes for the final metrics of evaluation and testing
    runWiseUsage = MetricUsage(
        Events.STARTED, Events.COMPLETED, Events.ITERATION_COMPLETED
    )

    # Setup metric tracking for validation and testing
    for ev, label in [(evaluator, "val"), (evaluator_test, "test")]:
        # Compute accuracy to be used in the metrics below
        if config["model"] == "prototypical":
            accuracy = Accuracy(
                device=device,
            )
        elif config["model"] == "relation":
            accuracy = RNFSAccuracy(
                device=device,
            )
        else:
            raise Exception("WAT")

        # Define tracked metrics
        metrics = {
            f"{label}_accuracy": accuracy,
            f"{label}_loss": Loss(
                loss_fn,
                device=device,
            ),
            f"{label}_error": (1.0 - accuracy) * 100,
        }
        # Attach to trainers
        for name, metric in metrics.items():
            metric.attach(ev, name, usage=runWiseUsage)
    # * Confusion matrices only for test evaluator
    Confusion_matrices().attach(
        evaluator_test, "test_confusion_matrices", usage=runWiseUsage
    )

    # Global metrics
    best_accuracy = 0
    best_test_accuracy = 0
    best_test_cf = []
    best_weights = {}
    best_epoch = 0
    patience_counter = 0  # FIXME Could patience be done with Ignite?

    # Setup logging
    train_logger = setup_logging(config, "trainer")
    train_logger.info("Configuration: \n%s", pformat(config))
    train_logger.info("Device: \n%s", str(device))
    train_logger.info("Model: \n%s", str(model))
    trainer.logger = train_logger
    evaluator.logger = setup_logging(config, "evaluator")
    evaluator_test.logger = setup_logging(config, "test")

    # Tensorboard logger

    # Only if logging turned on (make extra sure turned off for Optuna trials and result generation)
    if run_type == "normal" and config["log"]:
        abs_path = os.path.dirname(__file__)
        time = datetime.now().strftime("%m-%d_%H-%M-%S")

        tb_logger = TensorboardLogger(
            log_dir=os.path.join(
                abs_path,
                os.pardir,
                "reports",
                "RAW",
                "logs_tb",
                f"{config['name']}-{config['data']}-{config['backbone']}-{time}",
            )
        )

        # Log training loss after each batch
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            metric_names=["train_loss"],
        )

        # Log validation evaluator result
        # * A separate evaluator is created each time validation is run, thus EVENTS.COMPLETED
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.COMPLETED,
            tag="validation",
            metric_names=["val_accuracy", "val_loss"],
            global_step_transform=global_step_from_engine(trainer),
        )

        # Log test evaluator result
        tb_logger.attach_output_handler(
            evaluator_test,
            event_name=Events.COMPLETED,
            tag="test",
            metric_names=["test_accuracy", "test_loss"],
            global_step_transform=global_step_from_engine(trainer),
        )

        tb_logger.close()

    # Ignite handlers

    # Take learning rate scheduler step after every bach
    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def _():
        if scheduler is not None:
            scheduler.step()

    # For printing clarity
    @trainer.on(Events.EPOCH_STARTED)
    def _():
        print()

    # Print train metrics every epoch
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=1),
        log_metrics,
        tag="train",
    )

    # Run validation evaluation
    @trainer.on(Events.EPOCH_COMPLETED(every=config["eval_freq"]))
    def _():
        # Run validation evaluation
        evaluator.run(validation_loader, epoch_length=config["eval_episodes"])
        # Print validation results
        log_metrics(evaluator, "eval")

        # Save best accuracy and run test set
        # * The `nonlocal` is a bit of a hack, but it's the easiest way I found
        nonlocal best_accuracy
        nonlocal best_test_accuracy
        nonlocal best_test_cf
        nonlocal best_weights
        nonlocal best_epoch
        nonlocal patience_counter

        # If new best
        if evaluator.state.metrics["val_accuracy"] > best_accuracy:
            # Save current best validation result
            best_accuracy = evaluator.state.metrics["val_accuracy"]
            best_epoch = trainer.state.epoch

            # Reset patience
            patience_counter = 0

            # Run test evaluation only if validation accuracy has improved and not an Optuna trial
            if not run_type == "trial":
                evaluator_test.run(test_loader, epoch_length=config["test_episodes"])
                log_metrics(evaluator_test, "test")
                best_test_accuracy = evaluator_test.state.metrics["test_accuracy"]
                best_test_cf = evaluator_test.state.metrics["test_confusion_matrices"]

                if run_type == "result":
                    best_weights = model.state_dict()

            # No sense in continuing if validation accuracy 100%
            if evaluator.state.metrics["val_accuracy"] == 1:
                trainer.terminate()
        # If not best
        else:
            # Increment early stopping counter if over warmup period
            if trainer.state.epoch > 20:
                patience_counter += config["eval_freq"]
            # Early stopping if validation score has stopped improving
            if patience_counter >= config["patience"]:
                trainer.terminate()

    # Run test evaluation every once in a while in addition to when new best validation score is achieved
    @trainer.on(Events.EPOCH_COMPLETED(every=12))
    def _():
        # * Skip for trials and result generation
        if run_type == "normal":
            evaluator_test.run(test_loader, epoch_length=config["test_episodes"])
            # Print results
            log_metrics(evaluator_test, "test")

    # At the end of the run
    @trainer.on(Events.COMPLETED)
    def _():
        # Print global results
        print()
        print(
            "Best validation accuracy: {} @ epoch {}".format(best_accuracy, best_epoch)
        )
        print()
        print("Respective test accuracy: {}".format(best_test_accuracy))
        print()

    # Checkpoint system
    if config["save"]:
        base_dir_name = f"{config['name']}_{config['data']}_{config.get('job_id', datetime.now().strftime('%m-%d_%H-%M-%S'))}"
        checkpointer = ModelCheckpoint(
            dirname=f"./model_weights/{base_dir_name}/{config.get('i', 0)}_{datetime.now().strftime('%m-%d_%H-%M-%S')}",
            create_dir=True,
            n_saved=5,  # TODO Could use this for ensembles
            # filename_prefix="best",
            score_name="val_accuracy",
            global_step_transform=global_step_from_engine(trainer),
            filename_pattern="model_{score}.pth",
            require_empty=False,
            greater_or_equal=False,
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpointer, {"model": model})

    # TODO Checkpoint system
    # TODO Optuna pruning (This hasn't been working)
    # Start training
    trainer.run(
        train_loader, max_epochs=config["max_epochs"], epoch_length=config["epoch_len"]
    )

    # Used by optuna (best_accuracy) and result generation (best_test_accuracy, best_test_cf)
    return best_accuracy, best_test_accuracy, best_test_cf, best_weights


def setup_device():
    # Enable cuDNN autotuner (speed optimization)
    torch.backends.cudnn.benchmark = True

    # Set torch download directory (relevant for weights of pretrained networks)
    torch.hub.set_dir("./torch_downloads")

    # Setup device
    device_type = "cpu"
    if torch.cuda.is_available():
        device_type = "cuda"
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_built():
        device_type = "mps"
        # device_type = "cpu"

    device = torch.device(device_type)

    print()
    print("Using device:", device)
    # Additional info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
    print()

    return device


def run(config):
    # * Data-level loop
    # * Use this to group multiple runs with the same data (splits)
    # * Mostly used as a base for the hyperparameter optimization

    # INITIALIZATION
    ################

    device = setup_device()

    # DATA
    ######

    # Initialize data
    train_loader, validation_loader, test_loader = setup_data(config, device)

    # TRAIN
    #######

    run_training(
        train_loader,
        validation_loader,
        test_loader,
        config,
        device=device,
    )


if __name__ == "__main__":
    # Init arguments
    parser = ArgumentParser()
    parser.add_argument(f"--config", default="arotor_replication/base", type=type("a"))

    # Parse args
    args = parser.parse_args()

    # Read config
    config = setup_config(args.config)

    # Run setup
    run(config)
