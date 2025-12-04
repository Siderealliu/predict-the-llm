"""简单的统一日志工具，便于在各模块复用。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str, log_file: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    """创建并返回带文件输出的logger。"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
