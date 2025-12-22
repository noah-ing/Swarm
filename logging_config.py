"""Logging and observability for Swarm."""

import json
import logging
import sys
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


# Configure rich console
console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Set up logging for Swarm.

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

    return logger


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


@dataclass
class TaskMetrics:
    """Metrics for a task execution."""

    task_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    subtasks_total: int = 0
    subtasks_completed: int = 0
    subtasks_failed: int = 0
    retries: int = 0
    model_calls: list[dict] = field(default_factory=list)

    def complete(self) -> None:
        """Mark the task as complete."""
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def add_model_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record a model call."""
        self.model_calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "subtasks_total": self.subtasks_total,
            "subtasks_completed": self.subtasks_completed,
            "subtasks_failed": self.subtasks_failed,
            "retries": self.retries,
            "model_calls": self.model_calls,
        }


class MetricsCollector:
    """Collects and aggregates metrics across tasks."""

    def __init__(self, persist_path: str | None = None):
        self.tasks: list[TaskMetrics] = []
        self.persist_path = Path(persist_path) if persist_path else None

    def start_task(self, task_id: str) -> TaskMetrics:
        """Start tracking a new task."""
        metrics = TaskMetrics(task_id=task_id)
        self.tasks.append(metrics)
        return metrics

    def get_current(self) -> TaskMetrics | None:
        """Get the current (last) task metrics."""
        return self.tasks[-1] if self.tasks else None

    def get_totals(self) -> dict:
        """Get aggregate totals across all tasks."""
        return {
            "total_tasks": len(self.tasks),
            "total_input_tokens": sum(t.input_tokens for t in self.tasks),
            "total_output_tokens": sum(t.output_tokens for t in self.tasks),
            "total_cost_usd": sum(t.cost_usd for t in self.tasks),
            "total_duration_seconds": sum(t.duration_seconds for t in self.tasks),
            "total_retries": sum(t.retries for t in self.tasks),
        }

    def persist(self) -> None:
        """Persist metrics to disk."""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "tasks": [t.to_dict() for t in self.tasks],
            "totals": self.get_totals(),
            "exported_at": datetime.utcnow().isoformat(),
        }

        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load metrics from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        with open(self.persist_path) as f:
            data = json.load(f)

        # Reconstruct TaskMetrics objects
        for task_data in data.get("tasks", []):
            metrics = TaskMetrics(
                task_id=task_data["task_id"],
                input_tokens=task_data["input_tokens"],
                output_tokens=task_data["output_tokens"],
                cost_usd=task_data["cost_usd"],
                subtasks_total=task_data["subtasks_total"],
                subtasks_completed=task_data["subtasks_completed"],
                subtasks_failed=task_data["subtasks_failed"],
                retries=task_data["retries"],
            )
            metrics.duration_seconds = task_data["duration_seconds"]
            self.tasks.append(metrics)


# Global metrics collector
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector(persist_path: str | None = None) -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(persist_path)
    return _metrics_collector
