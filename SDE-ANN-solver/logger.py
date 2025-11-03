# coding=utf-8
import logging
import os
from datetime import datetime
from typing import Optional

from main import model


def get_logger(
    name: Optional[str] = None,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    to_file: bool = False,
    overwrite: bool = False,
) -> logging.Logger:
    """
    Creates and configures a logger for experiments or model training.

    Args:
        name: Optional[str]
            Name of the logger (e.g., 'train', 'eval'). If None, uses root logger.
        log_dir: Optional[str]
            Directory where log files are saved. If None, logs only to console.
        level: int
            Logging level (default=logging.INFO).
        to_file: bool
            Whether to log to a file in addition to the console.
        overwrite: bool
            If True, overwrites previous log file with the same name.
            If False, appends a timestamp suffix to avoid overwriting.

    Returns:
        logging.Logger
            Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers in notebooks / re-imports
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ---- Console handler ----
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- File handler ----
    if to_file and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_filename = f"{name or 'log'}.log"
        if not overwrite:
            log_filename = f"{name or 'log'}_{timestamp}.log"

        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    return logger


logger = get_logger(log_dir=".", to_file=True, overwrite=False)
