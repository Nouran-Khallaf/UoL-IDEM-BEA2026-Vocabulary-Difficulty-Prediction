from __future__ import annotations

import logging
from pathlib import Path


DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _normalize_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    if not isinstance(level, str) or not level.strip():
        raise ValueError(f"Invalid log level: {level!r}")

    level_name = level.strip().upper()
    if not hasattr(logging, level_name):
        raise ValueError(f"Unsupported log level: {level!r}")

    numeric_level = getattr(logging, level_name)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level!r}")

    return numeric_level


def _build_formatter(
    *,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Formatter:
    return logging.Formatter(fmt=log_format, datefmt=date_format)


def _has_console_handler(logger: logging.Logger) -> bool:
    return any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, *, level: str | int = "INFO") -> logging.Logger:
    """
    Return a configured logger with a console handler.
    Repeated calls do not duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    if not _has_console_handler(logger):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_normalize_level(level))
        console_handler.setFormatter(_build_formatter())
        logger.addHandler(console_handler)

    return logger


def configure_logger(
    name: str,
    *,
    level: str | int = "INFO",
    log_to_file: bool = False,
    log_file: str | Path | None = None,
    reset_handlers: bool = False,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    if reset_handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = _build_formatter(log_format=log_format, date_format=date_format)

    if not _has_console_handler(logger):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_normalize_level(level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_to_file:
        if log_file is None:
            raise ValueError("log_file must be provided when log_to_file=True")

        log_path = Path(log_file).resolve()
        _ensure_parent_dir(log_path)

        existing_file_handler = False
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                existing_path = getattr(handler, "baseFilename", None)
                if existing_path == str(log_path):
                    existing_file_handler = True
                    break

        if not existing_file_handler:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(_normalize_level(level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def configure_run_logger(
    run_dir: str | Path,
    *,
    logger_name: str = "bea_pipeline",
    level: str | int = "INFO",
    log_filename: str = "run.log",
    reset_handlers: bool = True,
) -> logging.Logger:
    """
    Configure a run-specific logger that writes to console and to a log file
    inside the run directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    return configure_logger(
        logger_name,
        level=level,
        log_to_file=True,
        log_file=run_dir / log_filename,
        reset_handlers=reset_handlers,
    )