"""Rich UI components for Swarm."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.style import Style
from rich.box import ROUNDED, HEAVY, DOUBLE

from agents.base import StreamEvent


console = Console()


# Color scheme
COLORS = {
    "primary": "bright_blue",
    "secondary": "bright_cyan",
    "success": "bright_green",
    "error": "bright_red",
    "warning": "bright_yellow",
    "muted": "dim white",
    "tool": "bright_magenta",
    "thinking": "bright_yellow",
}


def format_cost(cost: float) -> str:
    """Format cost in dollars."""
    if cost < 0.001:
        return f"${cost:.5f}"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.3f}"


def format_tokens(tokens: int) -> str:
    """Format token count."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}K"
    return str(tokens)


def format_duration(seconds: float) -> str:
    """Format duration."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


@dataclass
class GruntStatus:
    """Status of a grunt agent."""
    id: int
    task: str
    status: str = "pending"  # pending, running, completed, failed
    current_tool: str | None = None
    tool_input: str = ""
    thinking: str = ""
    result: str = ""
    start_time: float | None = None
    end_time: float | None = None
    retries: int = 0


class LiveDashboard:
    """Real-time dashboard for swarm execution."""

    def __init__(self, task: str, model: str):
        self.task = task
        self.model = model
        self.grunts: dict[int, GruntStatus] = {}
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = time.time()
        self.log_lines: list[str] = []
        self._live: Live | None = None

    def add_grunt(self, grunt_id: int, task: str):
        """Add a grunt to track."""
        self.grunts[grunt_id] = GruntStatus(
            id=grunt_id,
            task=task,
            start_time=time.time(),
        )

    def update_grunt(
        self,
        grunt_id: int,
        status: str | None = None,
        current_tool: str | None = None,
        tool_input: str | None = None,
        thinking: str | None = None,
        result: str | None = None,
        retries: int | None = None,
    ):
        """Update grunt status."""
        if grunt_id not in self.grunts:
            return

        grunt = self.grunts[grunt_id]
        if status:
            grunt.status = status
            if status in ("completed", "failed"):
                grunt.end_time = time.time()
        if current_tool is not None:
            grunt.current_tool = current_tool
        if tool_input is not None:
            grunt.tool_input = tool_input[:100]
        if thinking is not None:
            grunt.thinking = thinking[:200]
        if result is not None:
            grunt.result = result
        if retries is not None:
            grunt.retries = retries

        self._refresh()

    def log(self, message: str):
        """Add a log line."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[dim]{timestamp}[/dim] {message}")
        if len(self.log_lines) > 10:
            self.log_lines.pop(0)
        self._refresh()

    def update_stats(self, tokens: int = 0, cost: float = 0.0):
        """Update token/cost stats."""
        self.total_tokens += tokens
        self.total_cost += cost
        self._refresh()

    def _build_layout(self) -> Panel:
        """Build the dashboard layout."""
        # Header
        elapsed = time.time() - self.start_time
        header = Table.grid(padding=1)
        header.add_column(justify="left", ratio=3)
        header.add_column(justify="right", ratio=1)
        header.add_row(
            Text(f"Task: {self.task[:60]}{'...' if len(self.task) > 60 else ''}", style="bold"),
            Text(f"{self.model} | {format_duration(elapsed)}", style="dim"),
        )

        # Stats bar
        stats = Table.grid(padding=2)
        stats.add_column()
        stats.add_column()
        stats.add_column()
        stats.add_column()

        completed = sum(1 for g in self.grunts.values() if g.status == "completed")
        failed = sum(1 for g in self.grunts.values() if g.status == "failed")
        running = sum(1 for g in self.grunts.values() if g.status == "running")

        stats.add_row(
            Text(f"Grunts: {completed}/{len(self.grunts)}", style=COLORS["success"] if completed == len(self.grunts) else COLORS["primary"]),
            Text(f"Running: {running}", style=COLORS["secondary"] if running > 0 else "dim"),
            Text(f"Tokens: {format_tokens(self.total_tokens)}", style="dim"),
            Text(f"Cost: {format_cost(self.total_cost)}", style="dim"),
        )

        # Grunt table
        grunt_table = Table(box=ROUNDED, expand=True, show_header=True, header_style="bold")
        grunt_table.add_column("#", width=3, style="dim")
        grunt_table.add_column("Task", ratio=3)
        grunt_table.add_column("Status", width=12)
        grunt_table.add_column("Activity", ratio=2)

        for grunt in sorted(self.grunts.values(), key=lambda g: g.id):
            # Status indicator
            if grunt.status == "completed":
                status = Text("✓ done", style=COLORS["success"])
            elif grunt.status == "failed":
                status = Text("✗ fail", style=COLORS["error"])
            elif grunt.status == "running":
                status = Text("● run", style=COLORS["secondary"])
            else:
                status = Text("○ wait", style="dim")

            # Activity
            if grunt.current_tool:
                activity = Text(f"→ {grunt.current_tool}", style=COLORS["tool"])
                if grunt.tool_input:
                    activity.append(f" {grunt.tool_input[:30]}...", style="dim")
            elif grunt.thinking:
                activity = Text(f"💭 {grunt.thinking[:40]}...", style=COLORS["thinking"])
            elif grunt.status == "completed":
                activity = Text(grunt.result[:40] + "..." if len(grunt.result) > 40 else grunt.result, style="dim")
            else:
                activity = Text("-", style="dim")

            grunt_table.add_row(
                str(grunt.id),
                grunt.task[:50] + ("..." if len(grunt.task) > 50 else ""),
                status,
                activity,
            )

        # Log area
        log_text = "\n".join(self.log_lines[-5:]) if self.log_lines else "[dim]No activity yet...[/dim]"

        # Combine into layout
        content = Group(
            header,
            Text(""),
            stats,
            Text(""),
            grunt_table,
            Text(""),
            Text("Log:", style="bold dim"),
            Text.from_markup(log_text),
        )

        return Panel(
            content,
            title="[bold bright_blue]⬡ Swarm[/bold bright_blue]",
            border_style=COLORS["primary"],
            box=DOUBLE,
        )

    def _refresh(self):
        """Refresh the display."""
        if self._live:
            self._live.update(self._build_layout())

    def __enter__(self):
        """Start live display."""
        self._live = Live(
            self._build_layout(),
            console=console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        """Stop live display."""
        if self._live:
            self._live.__exit__(*args)


class StreamingDisplay:
    """Display for streaming agent output."""

    def __init__(self, title: str = "Agent"):
        self.title = title
        self.text_buffer = ""
        self.current_tool = None
        self.tool_buffer = ""
        self._live: Live | None = None

    def handle_event(self, event: StreamEvent):
        """Handle a stream event."""
        if event.type == "text":
            self.text_buffer += event.content
        elif event.type == "tool_start":
            self.current_tool = event.tool_name
            self.tool_buffer = ""
        elif event.type == "tool_input":
            self.tool_buffer += event.content
        elif event.type == "tool_end":
            self.current_tool = None
        elif event.type == "done":
            pass

        self._refresh()

    def _build_panel(self) -> Panel:
        """Build the display panel."""
        parts = []

        if self.text_buffer:
            parts.append(Text(self.text_buffer[-500:], style="white"))

        if self.current_tool:
            tool_display = Text(f"\n→ {self.current_tool}", style=COLORS["tool"])
            if self.tool_buffer:
                # Try to show as syntax
                try:
                    import json
                    parsed = json.loads(self.tool_buffer)
                    tool_display.append(f"\n{json.dumps(parsed, indent=2)[:200]}", style="dim")
                except:
                    tool_display.append(f"\n{self.tool_buffer[:200]}", style="dim")
            parts.append(tool_display)

        content = Group(*parts) if parts else Text("...", style="dim")

        return Panel(
            content,
            title=f"[bold]{self.title}[/bold]",
            border_style=COLORS["secondary"],
        )

    def _refresh(self):
        """Refresh display."""
        if self._live:
            self._live.update(self._build_panel())

    def __enter__(self):
        self._live = Live(
            self._build_panel(),
            console=console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)


def print_banner():
    """Print the Swarm banner."""
    banner = """
[bold bright_blue]
   ____
  / ___|_      ____ _ _ __ _ __ ___
  \\___ \\ \\ /\\ / / _` | '__| '_ ` _ \\
   ___) \\ V  V / (_| | |  | | | | | |
  |____/ \\_/\\_/ \\__,_|_|  |_| |_| |_|
[/bold bright_blue]
[dim]Multi-Agent Task Execution[/dim]
"""
    console.print(banner)


def print_result(success: bool, message: str, stats: dict | None = None):
    """Print the final result."""
    if success:
        panel = Panel(
            Markdown(message),
            title="[bold green]✓ Success[/bold green]",
            border_style="green",
        )
    else:
        panel = Panel(
            message,
            title="[bold red]✗ Failed[/bold red]",
            border_style="red",
        )

    console.print(panel)

    if stats:
        stat_text = " | ".join([
            f"Tokens: {format_tokens(stats.get('tokens', 0))}",
            f"Cost: {format_cost(stats.get('cost', 0))}",
            f"Time: {format_duration(stats.get('duration', 0))}",
        ])
        console.print(f"[dim]{stat_text}[/dim]")


def print_subtasks(subtasks: list[dict]):
    """Print subtask summary table."""
    table = Table(title="Subtasks", box=ROUNDED, show_header=True, header_style="bold")
    table.add_column("#", width=3, style="dim")
    table.add_column("Task", ratio=3)
    table.add_column("Status", width=12)
    table.add_column("Retries", width=8, justify="right")

    for st in subtasks:
        status_map = {
            "completed": ("[green]✓ done[/green]"),
            "failed": ("[red]✗ fail[/red]"),
            "running": ("[blue]● run[/blue]"),
            "pending": ("[dim]○ wait[/dim]"),
        }
        status = status_map.get(st.get("status", "pending"), st.get("status", ""))

        table.add_row(
            str(st.get("id", "?")),
            st.get("task", "")[:50] + ("..." if len(st.get("task", "")) > 50 else ""),
            status,
            str(st.get("retries", 0)) if st.get("retries", 0) > 0 else "-",
        )

    console.print(table)
