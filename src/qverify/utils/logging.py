"""
Logging Utilities for QVERIFY.

This module provides logging configuration and utilities for consistent
logging throughout the framework.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Custom log levels
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class QVerifyFormatter(logging.Formatter):
    """Custom formatter for QVERIFY logs."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'TRACE': '\033[90m',     # Gray
    }
    RESET = '\033[0m'
    
    def __init__(self, use_colors: bool = True) -> None:
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Level
        level = record.levelname
        if self.use_colors and level in self.COLORS:
            level = f"{self.COLORS[level]}{level:8}{self.RESET}"
        else:
            level = f"{level:8}"
        
        # Module
        module = record.name[-20:] if len(record.name) > 20 else record.name
        
        # Message
        message = record.getMessage()
        
        # Exception info
        if record.exc_info:
            message += '\n' + self.formatException(record.exc_info)
        
        return f"{timestamp} | {level} | {module:20} | {message}"


class QVerifyLogger:
    """
    Logger wrapper for QVERIFY.
    
    Example:
        >>> logger = get_logger("qverify.synthesis")
        >>> logger.info("Starting synthesis")
        >>> logger.debug("Processing program", extra={"program": "bell_state"})
    """
    
    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Set up default handlers if none exist."""
        if not self.logger.handlers:
            # Console handler
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(QVerifyFormatter(use_colors=True))
            self.logger.addHandler(console)
    
    def add_file_handler(self, filepath: Path, level: int = logging.DEBUG) -> None:
        """Add a file handler."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(filepath)
        handler.setLevel(level)
        handler.setFormatter(QVerifyFormatter(use_colors=False))
        self.logger.addHandler(handler)
    
    def trace(self, message: str, *args, **kwargs) -> None:
        """Log at TRACE level."""
        self.logger.log(TRACE, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log at DEBUG level."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log at INFO level."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log at WARNING level."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log at ERROR level."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log at CRITICAL level."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)


# Global loggers cache
_loggers: dict[str, QVerifyLogger] = {}


def get_logger(name: str, level: Optional[int] = None) -> QVerifyLogger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name (usually module name)
        level: Optional log level override
        
    Returns:
        QVerifyLogger instance
    """
    if name not in _loggers:
        _loggers[name] = QVerifyLogger(name, level or logging.INFO)
    elif level is not None:
        _loggers[name].logger.setLevel(level)
    
    return _loggers[name]


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure global logging settings.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Optional custom format string
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    root = logging.getLogger("qverify")
    root.setLevel(log_level)
    
    # Remove existing handlers
    root.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(QVerifyFormatter(use_colors=True))
    root.addHandler(console)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(QVerifyFormatter(use_colors=False))
        root.addHandler(file_handler)


class LogContext:
    """
    Context manager for temporary logging configuration.
    
    Example:
        >>> with LogContext(level="DEBUG"):
        ...     logger.debug("This will be logged")
        >>> logger.debug("This won't be logged at default INFO level")
    """
    
    def __init__(
        self,
        level: Optional[str] = None,
        logger_name: str = "qverify",
    ) -> None:
        self.level = level
        self.logger_name = logger_name
        self._previous_level: Optional[int] = None
    
    def __enter__(self) -> 'LogContext':
        if self.level:
            logger = logging.getLogger(self.logger_name)
            self._previous_level = logger.level
            logger.setLevel(getattr(logging, self.level.upper()))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._previous_level is not None:
            logger = logging.getLogger(self.logger_name)
            logger.setLevel(self._previous_level)


def log_function_call(logger: QVerifyLogger):
    """
    Decorator to log function calls.
    
    Example:
        >>> @log_function_call(logger)
        ... def synthesize(program):
        ...     ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator


# Default logger
default_logger = get_logger("qverify")
