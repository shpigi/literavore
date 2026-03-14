"""Logging utilities for literavore."""

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with structured format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=numeric_level, format=fmt, force=True)


def get_logger(name: str, stage: str | None = None) -> logging.Logger:
    """Return logger with optional stage context in format.

    When stage is provided, the log format includes the stage name:
        %(asctime)s [%(levelname)s] [<stage>] %(name)s: %(message)s
    Otherwise:
        %(asctime)s [%(levelname)s] %(name)s: %(message)s
    """
    logger = logging.getLogger(name)

    if stage is not None:
        handler = logging.StreamHandler()
        fmt = f"%(asctime)s [%(levelname)s] [{stage}] %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        # Avoid adding duplicate handlers if called multiple times with the same stage
        if not any(
            isinstance(h, logging.StreamHandler)
            and getattr(h.formatter, "_fmt", None) == fmt
            for h in logger.handlers
        ):
            logger.handlers.clear()
            logger.addHandler(handler)
            logger.propagate = False

    return logger
