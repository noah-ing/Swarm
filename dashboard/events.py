"""
Event system for capturing and broadcasting Swarm activity.

This module provides a central event bus that captures:
- Agent thoughts and reasoning
- Inter-agent messages
- Task execution progress
- Learning events
- Evolution mutations
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Callable, Any
from collections import deque
import threading


class EventType(Enum):
    # Agent lifecycle
    AGENT_SPAWN = "agent_spawn"
    AGENT_RETIRE = "agent_retire"

    # Thinking & reasoning
    THOUGHT = "thought"
    REASONING = "reasoning"
    STRATEGY = "strategy"

    # Inter-agent communication
    MESSAGE = "message"
    DIALOGUE = "dialogue"  # Agent-to-agent chat in dialogue room
    CRITIQUE = "critique"
    PROPOSAL = "proposal"
    CONSENSUS = "consensus"

    # Task execution
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"

    # Learning
    REFLECTION = "reflection"
    INSIGHT = "insight"
    SKILL_LEARNED = "skill_learned"

    # Evolution
    MUTATION = "mutation"
    VARIANT_CREATED = "variant_created"
    VARIANT_RETIRED = "variant_retired"

    # Goals
    GOAL_CREATED = "goal_created"
    GOAL_PROGRESS = "goal_progress"
    GOAL_COMPLETE = "goal_complete"

    # System
    STATS_UPDATE = "stats_update"


@dataclass
class SwarmEvent:
    """A single event in the Swarm system."""
    type: EventType
    timestamp: float = field(default_factory=time.time)

    # Source agent
    agent_id: str = ""
    agent_name: str = ""
    agent_role: str = ""

    # Target agent (for messages)
    target_agent_id: str = ""
    target_agent_name: str = ""

    # Content
    content: str = ""
    data: dict = field(default_factory=dict)

    # Metadata
    task_id: str = ""
    model: str = ""
    tokens: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["type"] = self.type.value
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class EventBus:
    """
    Central event bus for Swarm activity.

    Captures all events and broadcasts to subscribers (like the dashboard).
    """

    MAX_HISTORY = 1000  # Keep last N events

    def __init__(self):
        self._subscribers: list[Callable[[SwarmEvent], None]] = []
        self._async_subscribers: list[Callable[[SwarmEvent], Any]] = []
        self._history: deque[SwarmEvent] = deque(maxlen=self.MAX_HISTORY)
        self._lock = threading.Lock()

        # Track active agents
        self._active_agents: dict[str, dict] = {}

        # Track current tasks
        self._active_tasks: dict[str, dict] = {}

    def subscribe(self, callback: Callable[[SwarmEvent], None]):
        """Subscribe to events (sync callback)."""
        self._subscribers.append(callback)

    def subscribe_async(self, callback: Callable[[SwarmEvent], Any]):
        """Subscribe to events (async callback for WebSocket)."""
        self._async_subscribers.append(callback)

    def unsubscribe(self, callback):
        """Unsubscribe from events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
        if callback in self._async_subscribers:
            self._async_subscribers.remove(callback)

    def emit(self, event: SwarmEvent):
        """Emit an event to all subscribers."""
        with self._lock:
            self._history.append(event)

            # Track agents
            if event.type == EventType.AGENT_SPAWN:
                self._active_agents[event.agent_id] = {
                    "id": event.agent_id,
                    "name": event.agent_name,
                    "role": event.agent_role,
                    "spawned_at": event.timestamp,
                }
            elif event.type == EventType.AGENT_RETIRE:
                self._active_agents.pop(event.agent_id, None)

            # Track tasks
            if event.type == EventType.TASK_START:
                self._active_tasks[event.task_id] = {
                    "id": event.task_id,
                    "content": event.content,
                    "started_at": event.timestamp,
                    "agent": event.agent_name,
                }
            elif event.type in (EventType.TASK_COMPLETE, EventType.TASK_ERROR):
                self._active_tasks.pop(event.task_id, None)

        # Notify sync subscribers
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                print(f"Event subscriber error: {e}")

        # Notify async subscribers (thread-safe)
        for callback in self._async_subscribers:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(callback(event))
            except RuntimeError:
                # No event loop in this thread - use call_soon_threadsafe
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.call_soon_threadsafe(
                            lambda cb=callback, e=event: asyncio.ensure_future(cb(e))
                        )
                except Exception:
                    pass

    def get_history(self, limit: int = 100, event_types: list[EventType] = None) -> list[SwarmEvent]:
        """Get recent event history."""
        with self._lock:
            events = list(self._history)
            if event_types:
                events = [e for e in events if e.type in event_types]
            return events[-limit:]

    def get_active_agents(self) -> dict[str, dict]:
        """Get currently active agents."""
        with self._lock:
            return dict(self._active_agents)

    def get_active_tasks(self) -> dict[str, dict]:
        """Get currently running tasks."""
        with self._lock:
            return dict(self._active_tasks)

    # Convenience methods for common events

    def thought(self, agent_name: str, content: str, agent_id: str = "", model: str = ""):
        """Emit a thought/reasoning event."""
        self.emit(SwarmEvent(
            type=EventType.THOUGHT,
            agent_id=agent_id,
            agent_name=agent_name,
            content=content,
            model=model,
        ))

    def message(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        from_id: str = "",
        to_id: str = "",
    ):
        """Emit an inter-agent message."""
        self.emit(SwarmEvent(
            type=EventType.MESSAGE,
            agent_id=from_id,
            agent_name=from_agent,
            target_agent_id=to_id,
            target_agent_name=to_agent,
            content=content,
        ))

    def task_start(self, task_id: str, task: str, agent: str = "supervisor"):
        """Emit task start event."""
        self.emit(SwarmEvent(
            type=EventType.TASK_START,
            task_id=task_id,
            agent_name=agent,
            content=task,
        ))

    def task_complete(self, task_id: str, result: str, success: bool, tokens: int = 0):
        """Emit task completion event."""
        self.emit(SwarmEvent(
            type=EventType.TASK_COMPLETE,
            task_id=task_id,
            content=result,
            tokens=tokens,
            data={"success": success},
        ))

    def reflection(self, content: str, insights: list[str] = None):
        """Emit reflection event."""
        self.emit(SwarmEvent(
            type=EventType.REFLECTION,
            agent_name="brain",
            content=content,
            data={"insights": insights or []},
        ))

    def critique(self, critic: str, target: str, content: str, score: float = 0):
        """Emit critique event."""
        self.emit(SwarmEvent(
            type=EventType.CRITIQUE,
            agent_name=critic,
            target_agent_name=target,
            content=content,
            data={"score": score},
        ))

    def proposal(self, agent: str, content: str, task_id: str = ""):
        """Emit proposal event."""
        self.emit(SwarmEvent(
            type=EventType.PROPOSAL,
            agent_name=agent,
            content=content,
            task_id=task_id,
        ))

    def goal_progress(self, goal_id: str, description: str, completed: int, total: int):
        """Emit goal progress event."""
        self.emit(SwarmEvent(
            type=EventType.GOAL_PROGRESS,
            content=description,
            data={"goal_id": goal_id, "completed": completed, "total": total},
        ))


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
