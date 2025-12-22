"""Conversation history management for Swarm."""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TaskRecord:
    """Record of a completed task."""

    task_id: str
    task: str
    result: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    files_modified: list[str] = field(default_factory=list)
    subtasks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task": self.task,
            "result": self.result,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "files_modified": self.files_modified,
            "subtasks": self.subtasks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskRecord":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            task=data["task"],
            result=data["result"],
            success=data["success"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model=data.get("model", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            files_modified=data.get("files_modified", []),
            subtasks=data.get("subtasks", []),
        )


class ConversationHistory:
    """Manages conversation history for a session."""

    def __init__(
        self,
        session_id: str | None = None,
        history_dir: str | None = None,
    ):
        self.session_id = session_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.history_dir = Path(history_dir or os.path.expanduser("~/.swarm/history"))
        self.tasks: list[TaskRecord] = []
        self._task_counter = 0

    @property
    def session_file(self) -> Path:
        """Get the session file path."""
        return self.history_dir / f"{self.session_id}.json"

    def generate_task_id(self) -> str:
        """Generate a unique task ID."""
        self._task_counter += 1
        return f"{self.session_id}_{self._task_counter:04d}"

    def add_task(
        self,
        task: str,
        result: str,
        success: bool,
        **kwargs,
    ) -> TaskRecord:
        """
        Add a completed task to history.

        Args:
            task: The task that was executed
            result: The result/output
            success: Whether it succeeded
            **kwargs: Additional fields (model, tokens, cost, etc.)

        Returns:
            The created TaskRecord
        """
        record = TaskRecord(
            task_id=self.generate_task_id(),
            task=task,
            result=result,
            success=success,
            **kwargs,
        )
        self.tasks.append(record)
        return record

    def get_recent(self, n: int = 5) -> list[TaskRecord]:
        """Get the N most recent tasks."""
        return self.tasks[-n:]

    def get_context_summary(self, max_chars: int = 2000) -> str:
        """
        Get a summary of recent history for context.

        Args:
            max_chars: Maximum characters for the summary

        Returns:
            Formatted summary string
        """
        if not self.tasks:
            return ""

        parts = ["## Recent Tasks\n"]
        char_count = len(parts[0])

        for task in reversed(self.tasks[-5:]):
            entry = f"\n- [{task.timestamp.strftime('%H:%M')}] {task.task[:50]}..."
            if task.success:
                entry += " ✓"
            else:
                entry += " ✗"

            if char_count + len(entry) > max_chars:
                break

            parts.append(entry)
            char_count += len(entry)

        return "".join(parts)

    def save(self) -> None:
        """Save history to disk."""
        self.history_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "created_at": datetime.utcnow().isoformat(),
            "tasks": [t.to_dict() for t in self.tasks],
        }

        with open(self.session_file, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        """
        Load history from disk.

        Returns:
            True if loaded successfully
        """
        if not self.session_file.exists():
            return False

        try:
            with open(self.session_file) as f:
                data = json.load(f)

            self.tasks = [TaskRecord.from_dict(t) for t in data.get("tasks", [])]
            self._task_counter = len(self.tasks)
            return True

        except (json.JSONDecodeError, KeyError):
            return False

    @classmethod
    def list_sessions(cls, history_dir: str | None = None) -> list[dict]:
        """
        List available sessions.

        Args:
            history_dir: History directory path

        Returns:
            List of session info dicts
        """
        history_dir = Path(history_dir or os.path.expanduser("~/.swarm/history"))

        if not history_dir.exists():
            return []

        sessions = []
        for file in sorted(history_dir.glob("*.json"), reverse=True):
            try:
                with open(file) as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id", file.stem),
                    "created_at": data.get("created_at", ""),
                    "task_count": len(data.get("tasks", [])),
                    "file": str(file),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return sessions

    @classmethod
    def resume_session(
        cls,
        session_id: str,
        history_dir: str | None = None,
    ) -> "ConversationHistory":
        """
        Resume a previous session.

        Args:
            session_id: The session ID to resume
            history_dir: History directory path

        Returns:
            ConversationHistory with loaded data
        """
        history = cls(session_id=session_id, history_dir=history_dir)
        history.load()
        return history


# Global history instance
_current_history: ConversationHistory | None = None


def get_history(session_id: str | None = None) -> ConversationHistory:
    """Get or create the current history instance."""
    global _current_history
    if _current_history is None or (session_id and session_id != _current_history.session_id):
        _current_history = ConversationHistory(session_id=session_id)
    return _current_history
