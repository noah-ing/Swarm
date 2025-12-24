"""
Swarm Dashboard Server.

FastAPI server with WebSocket support for real-time visualization
of Swarm's multi-agent activity.

Run with: python3 -m dashboard.server
Or: uvicorn dashboard.server:app --reload --port 8420
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
from datetime import datetime

from dashboard.events import get_event_bus, SwarmEvent, EventType


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_event(self, event: SwarmEvent):
        """Broadcast a Swarm event to all clients."""
        await self.broadcast(event.to_json())


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Subscribe to event bus
    event_bus = get_event_bus()

    async def forward_event(event: SwarmEvent):
        await manager.broadcast_event(event)

    event_bus.subscribe_async(forward_event)

    print("\n" + "=" * 60)
    print("  SWARM DASHBOARD")
    print("=" * 60)
    print(f"  Open: http://localhost:8420")
    print("=" * 60 + "\n")

    yield

    # Cleanup
    event_bus.unsubscribe(forward_event)


app = FastAPI(
    title="Swarm Dashboard",
    description="Real-time visualization of Swarm multi-agent activity",
    lifespan=lifespan,
)

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Swarm Dashboard</h1><p>Static files not found.</p>")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    await manager.connect(websocket)

    # Send initial state
    event_bus = get_event_bus()
    initial_state = {
        "type": "initial_state",
        "active_agents": event_bus.get_active_agents(),
        "active_tasks": event_bus.get_active_tasks(),
        "recent_events": [e.to_dict() for e in event_bus.get_history(50)],
    }
    await websocket.send_text(json.dumps(initial_state))

    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            # Could handle commands from dashboard here
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/stats")
async def get_stats():
    """Get current Swarm statistics."""
    try:
        from brain import get_brain
        from evolution import get_evolution
        from memory import get_memory_store
        from knowledge import get_knowledge_store
        from agentfactory import get_agent_factory

        brain = get_brain()
        evolution = get_evolution()
        memory = get_memory_store()
        knowledge = get_knowledge_store()
        factory = get_agent_factory()

        return {
            "brain": brain.get_stats(),
            "evolution": evolution.get_stats(),
            "memory": memory.get_stats(),
            "knowledge": knowledge.get_stats(),
            "agent_factory": factory.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/agents")
async def get_agents():
    """Get all agent blueprints."""
    try:
        from agentfactory import get_agent_factory
        factory = get_agent_factory()

        agents = []
        import sqlite3
        conn = sqlite3.connect(factory.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, role, generation, status, tasks_completed,
                   tasks_failed, parent_id, description
            FROM blueprints ORDER BY created_at DESC
        """)
        for row in cursor.fetchall():
            agents.append({
                "id": row[0],
                "name": row[1],
                "role": row[2],
                "generation": row[3],
                "status": row[4],
                "tasks_completed": row[5],
                "tasks_failed": row[6],
                "parent_id": row[7],
                "description": row[8],
            })
        conn.close()

        return {"agents": agents}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/history")
async def get_history(limit: int = 100, event_type: str = None):
    """Get event history."""
    event_bus = get_event_bus()

    event_types = None
    if event_type:
        try:
            event_types = [EventType(event_type)]
        except ValueError:
            pass

    events = event_bus.get_history(limit, event_types)
    return {"events": [e.to_dict() for e in events]}


@app.post("/api/run")
async def run_task(request: dict):
    """Run a task through the Supervisor."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    task = request.get("task", "")
    if not task:
        return {"error": "No task provided"}

    try:
        from agents.supervisor import Supervisor

        def execute():
            supervisor = Supervisor(working_dir=".")
            result = supervisor.run(task=task, skip_qa=True)
            return {
                "success": result.success,
                "message": result.message[:500] if result.message else "",
                "strategy": result.strategy_used,
                "tokens": result.tokens_used,
                "duration": result.duration_seconds,
                "learnings": result.learnings,
            }

        # Run in thread to not block
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, execute)
            return result

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/reflections")
async def get_reflections(limit: int = 20):
    """Get recent reflections from brain."""
    try:
        from brain import get_brain
        brain = get_brain()

        import sqlite3
        conn = sqlite3.connect(brain.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT task, outcome, success, insights, model_used,
                   tokens_used, confidence, timestamp
            FROM reflections
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        reflections = []
        for row in cursor.fetchall():
            reflections.append({
                "task": row[0],
                "outcome": row[1][:200] if row[1] else "",
                "success": bool(row[2]),
                "insights": json.loads(row[3]) if row[3] else [],
                "model": row[4],
                "tokens": row[5],
                "confidence": row[6],
                "timestamp": row[7],
            })
        conn.close()

        return {"reflections": reflections}
    except Exception as e:
        return {"error": str(e)}


def run_server(port: int = 8420):
    """Run the dashboard server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_server()
