"""Modular logger factory with per-module rotating log files."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_LOG_DIR = Path("logs")


class ModuleLogger:
    """Per-module logger with rotating file output and optional console output.

    Wraps :class:`logging.Logger` and adds context-manager support so handlers
    are cleanly flushed and closed on exit.

    Parameters
    ----------
    name:
        Logger name — use ``__name__`` of the calling module.
    level:
        Minimum logging level (default: ``logging.DEBUG``).
    log_dir:
        Directory where log files are written (default: ``logs/``).
    console:
        Whether to also emit records to *stdout* (default: ``True``).
    """

    def __init__(
        self,
        name: str,
        level: int = logging.DEBUG,
        log_dir: Path = DEFAULT_LOG_DIR,
        console: bool = True,
    ) -> None:
        self._logger = _build_logger(name, level, log_dir, console)

    # ------------------------------------------------------------------
    # Logging API — delegates to the underlying Logger
    # ------------------------------------------------------------------

    def debug(self, msg: object, *args: object, **kwargs: object) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: object) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: object) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: object) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: object, *args: object, **kwargs: object) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: object, *args: object, **kwargs: object) -> None:
        self._logger.exception(msg, *args, **kwargs)

    @property
    def name(self) -> str:
        """Return the underlying logger name."""
        return self._logger.name

    # ------------------------------------------------------------------
    # Context manager — flushes/closes handlers on __exit__
    # ------------------------------------------------------------------

    def __enter__(self) -> ModuleLogger:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        for handler in self._logger.handlers[:]:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
        _registry.pop(self._logger.name, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    """Convert a dotted module name to a filesystem-safe string."""
    return name.replace(".", "_").replace("-", "_")


def _build_logger(name: str, level: int, log_dir: Path, console: bool) -> logging.Logger:
    """Create and configure a stdlib Logger for *name*."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — return cached instance

    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate output via root logger

    formatter = logging.Formatter(LOG_FORMAT)

    # -- File handler (daily rotating) -----------------------------------
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{_safe_name(name)}_{date_str}.log"

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y%m%d"
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # -- Optional console handler -----------------------------------------
    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_registry: dict[str, ModuleLogger] = {}


def get_module_logger(
    name: str,
    level: int = logging.DEBUG,
    log_dir: Path | None = None,
    console: bool = True,
) -> ModuleLogger:
    """Return a cached :class:`ModuleLogger` for *name*.

    Creates a new logger (and its rotating log file) on first call; subsequent
    calls with the same *name* return the cached instance regardless of other
    arguments.

    Parameters
    ----------
    name:
        Module name — pass ``__name__``.
    level:
        Minimum log level (default: ``logging.DEBUG``).
    log_dir:
        Directory for log files (default: ``logs/``).
    console:
        Emit records to *stdout* as well (default: ``True``).
    """
    if name not in _registry:
        _registry[name] = ModuleLogger(
            name=name,
            level=level,
            log_dir=log_dir or DEFAULT_LOG_DIR,
            console=console,
        )
    return _registry[name]
