from __future__ import annotations

import logging

from fxtsrcdet.utils.logger import CYAN, GREEN, RESET, ColorFormatter


def _record(message: str, level: int = logging.INFO) -> logging.LogRecord:
    """Build a log record for formatter tests."""

    return logging.LogRecord(
        name="test.fxtsrcdet.logger",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_color_formatter_uses_cyan_for_all_header_styles(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    formatter = ColorFormatter("[%(levelname)s] %(message)s", enable_color=True)

    star_text = formatter.format(_record("**** Welcome to FXTSRCDET! ****"))
    equal_text = formatter.format(_record("================================"))

    assert star_text.startswith(CYAN)
    assert star_text.endswith(RESET)
    assert equal_text.startswith(CYAN)
    assert equal_text.endswith(RESET)


def test_color_formatter_keeps_info_green_for_non_headers(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    formatter = ColorFormatter("[%(levelname)s] %(message)s", enable_color=True)

    text = formatter.format(_record("ordinary info message"))

    assert text.startswith(GREEN)
    assert text.endswith(RESET)


def test_color_formatter_disables_ansi_when_color_is_off(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    formatter = ColorFormatter("[%(levelname)s] %(message)s", enable_color=False)

    text = formatter.format(_record("**** Welcome to FXTSRCDET! ****"))

    assert text == "[INFO] **** Welcome to FXTSRCDET! ****"
