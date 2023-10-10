import logging
import os
from datetime import datetime

from ignite.utils import setup_logger


def setup_logging(config, phase):
    green = "\033[32m"
    reset = "\033[0m"

    abs_path = os.path.dirname(__file__)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger = setup_logger(
        name=f"{green}[ignite]{reset} {phase.upper()}",
        level=logging.INFO,
        format="%(name)s: %(message)s",
        filepath=os.path.join(
            abs_path,
            os.pardir,
            os.pardir,
            "reports",
            "RAW",
            "logs",
            f"{time}-training-info.log",
        ),
    )
    return logger


def log_metrics(engine, tag):
    metrics = {}
    for k, v in engine.state.metrics.items():
        if k != "test_confusion_matrices":
            metrics[k] = "{:.5g}".format(v)

    metrics_format = "{0} [{1}/{2}]: {3}".format(tag, engine.state.epoch, engine.state.iteration, metrics)
    engine.logger.info(metrics_format)
