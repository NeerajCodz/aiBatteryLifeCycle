"""
src.utils.logger
================
Project-wide structured logging configuration.

Design goals
------------
* Single call to ``get_logger(name)`` everywhere — no boilerplate per module.
* Console output uses a compact, human-readable format with log levels coloured.
* File output writes one JSON record per line for machine-parseable audit trails.
* Both handlers respect the LOG_LEVEL environment variable (default ``INFO``).
* The rotating file handler caps each log file at 10 MB with up to 5 back-ups.

Usage
-----
    from src.utils.logger import get_logger
    log = get_logger(__name__)

    log.info("Starting training", extra={"epoch": 1, "lr": 1e-3})
    log.warning("NaN loss detected")
    log.error("Model checkpoint missing", exc_info=True)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOG_DIR = _PROJECT_ROOT / "artifacts" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "battery_lifecycle.log"

# ── Log level (override via environment) ─────────────────────────────────────
_LEVEL_NAME: str = os.environ.get("LOG_LEVEL", "INFO").upper()
_LEVEL: int = getattr(logging, _LEVEL_NAME, logging.INFO)

# ── ANSI colour codes ─────────────────────────────────────────────────────────
_RESET = "\033[0m"
_COLOURS: dict[int, str] = {
    logging.DEBUG: "\033[36m",     # cyan
    logging.INFO: "\033[32m",      # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[35m",  # magenta
}


class _ColourFormatter(logging.Formatter):
    """Console formatter with level-based ANSI colour and compact layout.

    Format:
        2024-11-15 12:34:56.789  INFO     src.models.lstm  Training epoch 5
    """

    _FMT = "%(asctime)s  %(levelname)-8s %(name)-32s %(message)s"
    _DATE = "%Y-%m-%d %H:%M:%S"

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATE)
        self._use_colour = use_colour and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _COLOURS.get(record.levelno, "") if self._use_colour else ""
        reset = _RESET if self._use_colour else ""
        record.levelname = f"{colour}{record.levelname}{reset}"
        record.name = record.name.replace("src.", "").replace("api.", "")[:32]
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    """File formatter that emits one JSON object per log record.

    Each record includes:
    ``time``, ``level``, ``logger``, ``message``, ``module``,
    ``lineno``, and any extra key/value pairs attached via
    ``logging.LogRecord.__dict__``.
    """

    _RESERVED = frozenset(logging.LogRecord(
        "", 0, "", 0, "", (), None).__dict__.keys()
    ) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "module": record.module,
            "lineno": record.lineno,
        }
        # Attach any extra fields passed via `extra={...}`
        for key, val in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ── Root logger configuration (run once) ─────────────────────────────────────
def _configure_root() -> None:
    root = logging.getLogger()
    if root.handlers:          # already configured — do not add duplicate handlers
        return

    root.setLevel(_LEVEL)

    # --- Console handler ---
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(_LEVEL)
    console.setFormatter(_ColourFormatter())
    root.addHandler(console)

    # --- Rotating JSON file handler ---
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)            # always capture DEBUG to file
    file_handler.setFormatter(_JsonFormatter())
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("matplotlib", "PIL", "h5py", "urllib3", "httpx", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_root()


# ── Public API ────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a named logger that is already attached to the project handlers.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module, e.g.
        ``src.models.lstm`` or ``api.model_registry``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Example
    -------
    >>> from src.utils.logger import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Loaded %d batteries", 30)
    """
    return logging.getLogger(name)
