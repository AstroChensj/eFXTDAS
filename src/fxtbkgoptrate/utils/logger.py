"""Logging helpers for the background-threshold optimizer."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def emit(logger: logging.Logger | None, level: str, message: str) -> None:
    """Send one log message through the provided logger or a default logger."""
    if logger is None:
        logger = build_stream_logger("eFXTDAS.fxtbkgoptrate.default")
    getattr(logger, level.lower(), logger.info)(message)


def build_stream_logger(name: str, level: str | int = logging.INFO) -> logging.Logger:
    """Build or reuse a stream logger for CLI output."""
    resolved_level = getattr(logging, str(level).upper(), level) if isinstance(level, str) else int(level)
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(resolved_level)
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    handler.setLevel(resolved_level)
    logger.addHandler(handler)
    logger.setLevel(resolved_level)
    logger.propagate = False
    return logger


def build_file_logger(name: str, logfile: str | Path, level: str | int = logging.INFO) -> logging.Logger:
    """Build or reuse a file logger."""
    resolved_level = getattr(logging, str(level).upper(), level) if isinstance(level, str) else int(level)
    logger = logging.getLogger(name)
    log_path = Path(logfile).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            logger.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    handler.setLevel(resolved_level)
    logger.addHandler(handler)
    logger.setLevel(resolved_level)
    logger.propagate = False
    return logger


def build_cli_logger(name: str, level: str | None, logfile: str | Path | None = None) -> logging.Logger:
    """Build the CLI logger with stream and optional file handlers."""
    resolved_level = getattr(logging, str(level).upper(), logging.INFO) if level else logging.INFO
    logger = build_stream_logger(name, resolved_level)
    if logfile is not None:
        build_file_logger(name, logfile, resolved_level)
    return logger
