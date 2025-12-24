"""Logging and observability for Swarm."""

import json
import logging
import sys
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

# Import custom exceptions
from exceptions import APIError, SwarmBaseException

# Configure rich console
console = Console()

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Set up logging for Swarm with comprehensive error handling.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        json_format: Use JSON format for logs

    Returns:
        Configured logger
    """
    logger = logging.getLogger("swarm")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers = []

    if json_format:
        # JSON format for structured logging
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    else:
        # Rich format for human-readable output
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    # Add custom exception logging
    def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions with comprehensive logging."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Uncaught Exception",
            extra={
                "type": exc_type.__name__,
                "value": str(exc_value),
                "traceback": exc_traceback
            }
        )

    sys.excepthook = handle_unhandled_exception

    return logger

class JsonFormatter(logging.Formatter):
    """Enhanced JSON log formatter with extensive error context."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Special handling for exceptions
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            log_data.update({
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value),
            })

        # Handle custom exception types
        if isinstance(record.msg, (APIError, SwarmBaseException)):
            log_data.update({
                "error_context": record.msg.context
            })

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)

@dataclass
class MetricsCollector:
    """Simple metrics collector stub."""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    def record_task(self, success: bool, tokens: int = 0, cost: float = 0.0):
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        self.total_tokens += tokens
        self.total_cost += cost

    def get_summary(self) -> dict:
        return asdict(self)


_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector