from __future__ import annotations

import logging
from pathlib import Path


def create_run_logger(run_dir: Path, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(f"ninfer_eval.{run_dir.name}")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S%z"
        )
    )
    logger.addHandler(handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
