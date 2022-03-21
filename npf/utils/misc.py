from typing import Optional

import os
import random
import logging
from datetime import datetime

from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

from .logging import RichHandler, RichFileHandler


__all__ = [
    "get_experiment_name",
    "setup_logger",
    "track",
]


_LOG_SHORT_FORMAT = "[%(asctime)s] %(message)s"
_LOG_LONG_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
_LOG_DATE_SHORT_FORMAT = "%H:%M:%S"
_LOG_DATE_LONG_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_experiment_name(random_code: Optional[str] = None) -> str:
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    if random_code is None:
        random_code = "".join(random.choices("abcdefghikmnopqrstuvwxyz", k=4))
    return  now + "-" + random_code


def setup_logger(logger_name: str, output_dir: str):
    logging.getLogger().addHandler(logging.NullHandler())
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    stream_handler = RichHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(fmt=_LOG_SHORT_FORMAT, datefmt=_LOG_DATE_SHORT_FORMAT))
    logger.addHandler(stream_handler)

    debug_file_handler = RichFileHandler(os.path.join(output_dir, "debug.log"), mode="w")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(logging.Formatter(fmt=_LOG_LONG_FORMAT, datefmt=_LOG_DATE_LONG_FORMAT))
    logger.addHandler(debug_file_handler)

    info_file_handler = RichFileHandler(os.path.join(output_dir, "info.log"), mode="w")
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(logging.Formatter(fmt=_LOG_SHORT_FORMAT, datefmt=_LOG_DATE_LONG_FORMAT))
    logger.addHandler(info_file_handler)

    return logger


def track(
    sequence,
    total = None,
    task_id = None,
    description: str = "Working...",
    update_period: float = 0.1,
):
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        SpinnerColumn(),
    ) as progress:
        for args in progress.track(
            sequence, total=total, task_id=task_id,
            description=description, update_period=update_period
        ):
            yield args
