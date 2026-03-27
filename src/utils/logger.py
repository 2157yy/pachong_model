"""
统一日志入口。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Set

LOGGER_NAME = "pachong_model"
_base_logger = logging.getLogger(LOGGER_NAME)
_configured_paths: Set[str] = set()


class BraceStyleLogger:
    """兼容 loguru 风格 logger.info("x {}", value) 的轻量封装。"""

    def __init__(self, base_logger: logging.Logger):
        self._base_logger = base_logger

    def _format(self, message: str, *args: object) -> str:
        if args:
            try:
                return message.format(*args)
            except Exception:
                return f"{message} {' '.join(map(str, args))}"
        return message

    def debug(self, message: str, *args: object) -> None:
        self._base_logger.debug(self._format(message, *args))

    def info(self, message: str, *args: object) -> None:
        self._base_logger.info(self._format(message, *args))

    def warning(self, message: str, *args: object) -> None:
        self._base_logger.warning(self._format(message, *args))

    def error(self, message: str, *args: object) -> None:
        self._base_logger.error(self._format(message, *args))

    def exception(self, message: str, *args: object) -> None:
        self._base_logger.exception(self._format(message, *args))


logger = BraceStyleLogger(_base_logger)


def setup_logger(log_dir: str | Path = "logs", file_prefix: str = "app", level: str = "INFO") -> BraceStyleLogger:
    """初始化控制台和文件日志，重复调用不会重复挂载。"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _base_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    _base_logger.propagate = False

    if not any(isinstance(handler, logging.StreamHandler) for handler in _base_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        _base_logger.addHandler(console_handler)

    log_path = str(log_dir / f"{file_prefix}.log")
    if log_path not in _configured_paths:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        _base_logger.addHandler(file_handler)
        _configured_paths.add(log_path)

    return logger
