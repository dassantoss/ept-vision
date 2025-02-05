#!/usr/bin/env python3
"""
Logging configuration module.

This module provides custom logging functionality with colored console output
and rotating file handlers. It includes a custom formatter for colored logs
and utility functions for logger setup and retrieval.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class CustomFormatter(logging.Formatter):
    """
    Custom formatter with colors for different log levels.

    This formatter applies different colors to log messages based on their
    level, making it easier to distinguish between different types of logs
    in the console output.

    Attributes:
        grey: Color code for debug messages
        blue: Color code for info messages
        yellow: Color code for warning messages
        red: Color code for error messages
        bold_red: Color code for critical messages
        reset: Code to reset text color
        format_str: Base format string for log messages
        FORMATS: Mapping of log levels to their colored format strings
    """

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with appropriate colors.

        Args:
            record: Log record to format

        Returns:
            str: Formatted and colored log message
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with console and optionally file output.

    This function creates a new logger with the specified name and
    configures it with a colored console handler and optionally a
    rotating file handler.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        max_size: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

    return logger


# Create default logger
logger = setup_logger(
    "ept_vision",
    log_file="logs/ept_vision.log"
)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This function creates a new logger with the given name and
    configures it with both console and file output.

    Args:
        name: Name for the logger, will be prefixed with 'ept_vision.'

    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logger(
        f"ept_vision.{name}",
        log_file=f"logs/{name}.log"
    )
