import logging
from pathlib import Path


def get_logger() -> logging.Logger:
    logger = logging.getLogger("ppe-api")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    Path("logs").mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler("logs/predictions.log")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
