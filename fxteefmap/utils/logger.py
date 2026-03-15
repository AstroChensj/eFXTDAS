"""Logging helpers for FXT EEF-map generation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD_WHITE = "\033[1;37m"


def _is_header_message(message: str) -> bool:
    """Return whether a message looks like a banner or section header."""
    stripped = message.strip()
    return (
        stripped.startswith("====")
        or stripped.endswith("====")
        or stripped.startswith("****")
        or stripped.endswith("****")
    )


class ColorFormatter(logging.Formatter):
    """ANSI-colored formatter for interactive terminal logs."""

    def __init__(self, fmt: str, *, enable_color: bool) -> None:
        super().__init__(fmt)
        self.enable_color = enable_color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self.enable_color or not os.getenv("TERM") or os.getenv("NO_COLOR"):
            return message
        if _is_header_message(record.getMessage()):
            color = CYAN if "=" in record.getMessage() else BOLD_WHITE
        else:
            color = {
                "INFO": GREEN,
                "WARNING": YELLOW,
                "ERROR": RED,
                "CRITICAL": RED,
            }.get(record.levelname, "")
        return f"{color}{message}{RESET}" if color else message


def _ensure_handler(
    logger: logging.Logger,
    handler: logging.Handler,
    *,
    level: int,
    fmt: str = "[%(levelname)s] %(message)s",
) -> logging.Logger:
    """Attach one configured handler if an equivalent one is not already present."""
    handler.setLevel(level)
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        stream = getattr(handler, "stream", None)
        enable_color = bool(getattr(stream, "isatty", lambda: False)())
        handler.setFormatter(ColorFormatter(fmt, enable_color=enable_color))
    else:
        handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _default_logger() -> logging.Logger:
    """Return a process-local default stream logger for fxteefmap."""
    logger = logging.getLogger("eFXTDAS.fxteefmap.default")
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
