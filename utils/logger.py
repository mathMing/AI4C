# -*- coding: utf-8 -*-
"""Logger and AverageMeter."""
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path


def get_logger(name: str, log_dir: Path = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    """滑动平均计数器。"""

    def __init__(self, name: str, window: int = 50):
        self.name = name
        self.values = deque(maxlen=window)
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.values.append(val)
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def __str__(self) -> str:
        return f"{self.name}={self.avg:.4f}"


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def __str__(self) -> str:
        return f"{self.elapsed():.1f}s"
