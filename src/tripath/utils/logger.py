from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
DEFAULT_LOG_DIR = Path("artifacts/logs")
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "docureason.log"


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logger(
    name: str = "docureason",
    log_level: Optional[str | int] = None,
    log_file: Optional[str | Path] = None,
    json_format: bool = False,
) -> logging.Logger:
    """Configure and return a logger with standard stream and optional file handlers.

    Parameters
    ----------
    name : str
        Logger module or root name.
    log_level : str or int, optional
        Log level (e.g. "DEBUG", "INFO", "WARNING", "ERROR"). Defaults to env
        var DOCUREASON_LOG_LEVEL or INFO.
    log_file : str or Path, optional
        Path to output log file. Defaults to env var DOCUREASON_LOG_FILE or
        `artifacts/logs/docureason.log`. Pass False or empty string to disable file logging.
    json_format : bool
        If True, log records as JSON objects.
    """
    if log_level is None:
        log_level_str = os.getenv("DOCUREASON_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, log_level_str, logging.INFO)
    elif isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = log_level

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check environment variable overrides for json formatting
    if os.getenv("DOCUREASON_LOG_FORMAT", "").lower() == "json":
        json_format = True

    formatter: logging.Formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Check if handlers already exist to avoid duplicate logs
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Setup File Handler if requested or by default
    if log_file is None:
        env_file = os.getenv("DOCUREASON_LOG_FILE")
        if env_file:
            log_file = Path(env_file)
        elif log_file is not False:
            log_file = DEFAULT_LOG_FILE

    if log_file:
        file_path = Path(log_file)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # Check if file handler already added for this path
            has_file_handler = any(
                isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == file_path.resolve()
                for h in logger.handlers
            )
            if not has_file_handler:
                file_handler = logging.FileHandler(file_path, encoding="utf-8")
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        except Exception as err:
            logger.warning("Could not set up file log handler for %s: %s", log_file, err)

    return logger


def get_logger(name: str = "docureason") -> logging.Logger:
    """Retrieve an existing logger or create one with default settings."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def log_pipeline_flag(
    flag_name: str,
    value: Any,
    reason: str = "",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log pipeline flag/feature switch decisions for traceability."""
    target_logger = logger or get_logger("docureason.pipeline")
    msg = f"[PIPELINE_FLAG] {flag_name} = {value}"
    if reason:
        msg += f" (Reason: {reason})"
    target_logger.info(msg)
