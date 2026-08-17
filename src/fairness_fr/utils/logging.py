"""Centralized logging configuration for the fairness evaluation pipeline.

Every long-running stage (embedding extraction over tens of thousands of
images, pair generation, metric computation) should log through this
module rather than calling ``print`` or configuring its own handlers.
Configuration is applied exactly once per process via :func:`setup_logging`;
subsequent calls are no-ops unless ``force`` is set, which keeps behavior
predictable when many modules import this file.

Note on the module name: this file is ``fairness_fr.logging``, not the
standard library ``logging`` module. Because Python 3 imports are absolute
by default, the ``import logging`` statement below resolves to the
standard library module, not to this file — this module simply wraps it.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Guards against re-configuring handlers if setup_logging is called
#: multiple times across different entrypoints/notebooks in one process.
_IS_CONFIGURED: bool = False


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
) -> None:
    """Configure the root logger for the whole pipeline.

    Args:
        level: Logging level name, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        log_file: Optional path to a log file. If provided, its parent
            directory is created if missing and a rotating file handler
            is attached alongside the console handler. Useful for large
            batch jobs (e.g. embedding extraction over a full dataset)
            where console output alone is easy to lose.
        max_bytes: Maximum size in bytes of a single log file before it
            is rotated. Only used when ``log_file`` is provided.
        backup_count: Number of rotated log files to retain.
        force: If True, reconfigure the root logger even if
            :func:`setup_logging` has already run in this process.

    Returns:
        None.
    """
    global _IS_CONFIGURED

    if _IS_CONFIGURED and not force:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Clear any pre-existing handlers so repeated `force=True` calls
    # (e.g. in notebooks re-running a setup cell) don't duplicate log lines.
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Quiet noisy third-party libraries that would otherwise drown out
    # pipeline logs during large-scale embedding extraction.
    for noisy_logger in ("PIL", "matplotlib", "urllib3", "onnxruntime"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _IS_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging with defaults if needed.

    Args:
        name: Logger name — conventionally ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    if not _IS_CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
