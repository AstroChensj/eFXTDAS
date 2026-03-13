"""Logging helpers for FXT source detection."""

from __future__ import annotations

import logging
from pathlib import Path


def _ensure_handler(
    logger: logging.Logger,
    handler: logging.Handler,
    *,
    level: int,
    fmt: str = "[%(levelname)s] %(message)s",
) -> logging.Logger:
    """Attach one configured handler if an equivalent one is not already present."""
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _default_logger() -> logging.Logger:
    """Return a process-local default stream logger for fxtsrcdet."""
    logger = logging.getLogger("eFXTDAS.fxtsrcdet.default")
    if logger.handlers:
        return logger
    return _ensure_handler(logger, logging.StreamHandler(), level=logging.INFO)


def emit(logger: logging.Logger | None, level: str, message: str) -> None:
    """Send a message to a logger or to the default stream logger."""
    if logger is None:
        logger = _default_logger()
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)


def build_stream_logger(name: str, level: str | int = logging.INFO) -> logging.Logger:
    """Create or reuse a stream logger for CLI use."""
    resolved_level = getattr(logging, str(level).upper(), level) if isinstance(level, str) else int(level)
    logger = logging.getLogger(name)
    if any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        logger.setLevel(resolved_level)
        return logger
    return _ensure_handler(logger, logging.StreamHandler(), level=resolved_level)


def build_file_logger(name: str, logfile: str | Path, level: str | int = logging.INFO) -> logging.Logger:
    """Create or reuse a file logger."""
    resolved_level = getattr(logging, str(level).upper(), level) if isinstance(level, str) else int(level)
    log_path = Path(logfile).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            logger.setLevel(resolved_level)
            return logger
    return _ensure_handler(logger, logging.FileHandler(log_path, mode="a"), level=resolved_level)


def build_cli_logger(name: str, level: str | None, logfile: str | Path | None = None) -> logging.Logger:
    """Create a CLI logger with optional stream and file handlers."""
    resolved_level = getattr(logging, str(level).upper(), logging.INFO) if level else logging.INFO
    logger = build_stream_logger(name, resolved_level)
    if logfile is not None:
        build_file_logger(name, logfile, resolved_level)
    return logger
