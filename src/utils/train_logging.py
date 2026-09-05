"""Plain UTF-8 logging for training subprocess (web log panel)."""
from __future__ import annotations

import re
import sys

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def configure_train_stdio() -> None:
    """Force UTF-8 stdout/stderr and disable loguru ANSI colors."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    from loguru import logger

    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        colorize=False,
        level="INFO",
    )