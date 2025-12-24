"""
Swarm Dashboard - Real-time visualization of multi-agent activity.

Usage:
    python3 -m dashboard.server

Then open http://localhost:8420 in your browser.
"""

from .events import EventBus, SwarmEvent, EventType, get_event_bus
from .server import app, run_server

__all__ = [
    "EventBus",
    "SwarmEvent",
    "EventType",
    "get_event_bus",
    "app",
    "run_server",
]
