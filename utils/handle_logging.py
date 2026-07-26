import logging
import os
import sys
from pathlib import Path

LOG_DIR = "logs"
LOG_FILE = "app"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(file_path: str):
    filename = Path(file_path).stem
    logger = logging.getLogger(filename)

    if not logger.handlers:
        log_file = os.path.join(LOG_DIR, f"{LOG_FILE}.log")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.stream.reconfigure(encoding="utf-8", errors="replace")
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )

        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8", errors="replace")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.setLevel(LOG_LEVEL)

    return logger
