#!/usr/bin/env python3
"""Swarm: A multi-agent CLI for autonomous task execution."""

import os
import sys
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import get_settings
from agents import GruntAgent, Orchestrator
from agents.base import StreamEvent
from router import ModelRouter
from history import ConversationHistory, get_history
from logging_config import setup_logging, get_metrics_collector
from context import ContextBuilder
from memory import get_memory_store
from ui import (
    console, print_banner, print_result, print_subtasks,
    format_cost, format_tokens, format_duration,
    LiveDashboard, StreamingDisplay, COLORS
)


@click.group(invoke_without_command=True)
@click.argument("task", required=False)
@click.option("--prefer", "-p", type=click.Choice(["anthropic", "openai"]), help="Preferred provider")
@click.option("--model", "-m", help="Force a specific model (haiku, sonnet, opus, gpt-4o-mini, gpt-4o)")
@click.option("--cheap", is_flag=True, help="Use cheapest models (haiku/gpt-4o-mini)")
@click.option("--single", "-s", is_flag=True, help="Single grunt mode (no orchestration)")
@click.option("--no-qa", is_flag=True, help="Skip QA validation")
@click.option("--no-parallel", is_flag=True, help="Disable parallel execution")
@click.option("--stream", is_flag=True, help="Stream agent output in real-time")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--working-dir", "-w", type=click.Path(exists=True), help="Working directory")
@click.option("--session", help="Resume a session by ID")
@click.pass_context
def main(
    ctx,
    task: str | None,
    prefer: str | None,
    model: str | None,
    cheap: bool,
    single: bool,
    no_qa: bool,
    no_parallel: bool,
    stream: bool,
    interactive: bool,
    verbose: bool,
    working_dir: str | None,
    session: str | None,
):
    """
    Swarm: Autonomous multi-agent task execution.

    \b
    Examples:
        swarm "create a python script that downloads a webpage"
        swarm "fix the bug in src/parser.py" --single
        swarm "refactor the authentication module" --prefer anthropic
        swarm --stream "explain main.py"
        swarm -i  # Interactive mode

    \b
    Commands:
        swarm history    List previous sessions
        swarm stats      Show usage statistics
        swarm memory     Show memory stats
    """
    if ctx.invoked_subcommand is not None:
        return

    settings = get_settings()

    if verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="INFO")

    # Check API keys
    if not settings.anthropic_api_key and not settings.openai_api_key:
        console.print("[red]Error: No API keys configured.[/red]")
        console.print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment or .env file.")
        sys.exit(1)

    # Initialize history
    history = get_history(session_id=session)
    if session:
        if history.load():
            console.print(f"[dim]Resumed session: {session} ({len(history.tasks)} prior tasks)[/dim]")
        else:
            console.print(f"[yellow]Warning: Could not load session {session}[/yellow]")

    working_dir = working_dir or os.getcwd()

    # Interactive mode
    if interactive or not task:
        run_interactive(
            prefer=prefer,
            model=model,
            cheap=cheap,
            single=single,
            no_qa=no_qa,
            no_parallel=no_parallel,
            stream=stream,
            verbose=verbose,
            working_dir=working_dir,
            history=history,
        )
        return

    # Single task mode
    run_task(
        task=task,
        prefer=prefer,
        model=model,
        cheap=cheap,
        single=single,
        no_qa=no_qa,
        no_parallel=no_parallel,
        stream=stream,
        verbose=verbose,
        working_dir=working_dir,
        history=history,
    )


def run_interactive(
    prefer: str | None,
    model: str | None,
    cheap: bool,
    single: bool,
    no_qa: bool,
    no_parallel: bool,
    stream: bool,
    verbose: bool,
    working_dir: str,
    history: ConversationHistory,
):
    """Run in interactive mode."""
    print_banner()
    console.print(f"[dim]Session: {history.session_id} | Working dir: {working_dir}[/dim]")
    console.print("[dim]Type your task, or 'quit' to exit. Commands: /history, /stats, /memory, /clear[/dim]\n")

    while True:
        try:
            task = console.input(f"[bold {COLORS['primary']}]swarm>[/bold {COLORS['primary']}] ").strip()
            if not task:
                continue

            if task.lower() in ("quit", "exit", "q"):
                history.save()
                console.print(f"[dim]Session saved: {history.session_id}[/dim]")
                break

            if task.startswith("/"):
                handle_command(task, history)
                continue

            run_task(
                task=task,
                prefer=prefer,
                model=model,
                cheap=cheap,
                single=single,
                no_qa=no_qa,
                no_parallel=no_parallel,
                stream=stream,
                verbose=verbose,
                working_dir=working_dir,
                history=history,
            )
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'quit' to exit.[/dim]")
            continue
        except EOFError:
            history.save()
            break


def handle_command(command: str, history: ConversationHistory):
    """Handle interactive commands."""
    cmd = command.lower().strip()

    if cmd == "/history":
        if not history.tasks:
            console.print("[dim]No tasks in this session yet.[/dim]")
            return

        table = Table(title=f"Session: {history.session_id}")
        table.add_column("Time", style="dim")
        table.add_column("Task", max_width=50)
        table.add_column("Status")
        table.add_column("Cost", justify="right")

        for task in history.tasks[-10:]:
            status = f"[{COLORS['success']}]✓[/{COLORS['success']}]" if task.success else f"[{COLORS['error']}]✗[/{COLORS['error']}]"
            table.add_row(
                task.timestamp.strftime("%H:%M:%S"),
                task.task[:50] + ("..." if len(task.task) > 50 else ""),
                status,
                format_cost(task.cost_usd),
            )

        console.print(table)

    elif cmd == "/stats":
        if not history.tasks:
            console.print("[dim]No tasks in this session yet.[/dim]")
            return

        total_cost = sum(t.cost_usd for t in history.tasks)
        total_input = sum(t.input_tokens for t in history.tasks)
        total_output = sum(t.output_tokens for t in history.tasks)
        success_rate = sum(1 for t in history.tasks if t.success) / len(history.tasks)

        console.print(Panel(
            f"Tasks: {len(history.tasks)}\n"
            f"Success rate: {success_rate:.0%}\n"
            f"Tokens: {format_tokens(total_input)} in / {format_tokens(total_output)} out\n"
            f"Total cost: {format_cost(total_cost)}",
            title="Session Stats",
            border_style=COLORS["primary"],
        ))

    elif cmd == "/memory":
        memory = get_memory_store()
        stats = memory.get_stats()
        model_stats = memory.get_model_stats()

        console.print(Panel(
            f"Total memories: {stats['total_memories']}\n"
            f"Success rate: {stats['success_rate']:.0%}\n"
            f"Total cost: {format_cost(stats['total_cost_usd'])}\n"
            f"Skills learned: {stats['total_skills']}",
            title="Memory Stats",
            border_style=COLORS["secondary"],
        ))

        if model_stats:
            table = Table(title="Model Performance")
            table.add_column("Model")
            table.add_column("Tasks", justify="right")
            table.add_column("Success", justify="right")
            table.add_column("Avg Tokens", justify="right")

            for model, data in model_stats.items():
                table.add_row(
                    model,
                    str(data["total_tasks"]),
                    f"{data['success_rate']:.0%}",
                    format_tokens(int(data["avg_tokens"] or 0)),
                )
            console.print(table)

    elif cmd == "/clear":
        console.clear()

    elif cmd == "/help":
        console.print(Panel(
            "/history - Show task history\n"
            "/stats   - Show session statistics\n"
            "/memory  - Show memory & model stats\n"
            "/clear   - Clear screen\n"
            "/help    - Show this help\n"
            "quit     - Exit and save session",
            title="Commands",
            border_style="dim",
        ))

    else:
        console.print(f"[yellow]Unknown command: {command}[/yellow]")


def run_task(
    task: str,
    prefer: str | None,
    model: str | None,
    cheap: bool,
    single: bool,
    no_qa: bool,
    no_parallel: bool,
    stream: bool,
    verbose: bool,
    working_dir: str,
    history: ConversationHistory,
):
    """Execute a single task."""
    settings = get_settings()
    metrics = get_metrics_collector()
    task_metrics = metrics.start_task(history.generate_task_id())
    memory = get_memory_store()
    start_time = time.time()

    # Determine model
    if cheap:
        model = "haiku" if (prefer or settings.prefer_provider) == "anthropic" else "gpt-4o-mini"
    elif not model:
        router = ModelRouter(prefer_provider=prefer)
        model = router.select(task)

    console.print(f"\n[dim]Model: {model} | Dir: {working_dir}[/dim]")

    # Get history context
    history_context = history.get_context_summary()

    if single:
        # Single grunt mode
        result = run_single_grunt(
            task=task,
            model=model,
            context=history_context,
            working_dir=working_dir,
            stream=stream,
            skip_qa=no_qa,
        )

        duration = time.time() - start_time

        if result["success"]:
            print_result(True, result["message"], {
                "tokens": result["input_tokens"] + result["output_tokens"],
                "cost": result["cost"],
                "duration": duration,
            })

            if result.get("files_modified"):
                console.print(f"[dim]Files: {', '.join(result['files_modified'])}[/dim]")
        else:
            print_result(False, result["error"] or "Unknown error", {
                "tokens": result["input_tokens"] + result["output_tokens"],
                "cost": result["cost"],
                "duration": duration,
            })

        # Track in memory
        memory.track_model_performance(
            model=model,
            task_type="single",
            success=result["success"],
            tokens_used=result["input_tokens"] + result["output_tokens"],
            cost_usd=result["cost"],
            duration_ms=int(duration * 1000),
        )

        # Save to history
        history.add_task(
            task=task,
            result=result["message"] if result["success"] else (result["error"] or "Failed"),
            success=result["success"],
            model=model,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=result["cost"],
            files_modified=result.get("files_modified", []),
        )

    else:
        # Orchestrated mode
        result = run_orchestrated(
            task=task,
            model=model,
            context=history_context,
            working_dir=working_dir,
            stream=stream,
            skip_qa=no_qa,
            parallel=not no_parallel,
        )

        duration = time.time() - start_time

        if result["subtasks"]:
            print_subtasks(result["subtasks"])

        if result["success"]:
            print_result(True, result["message"], {
                "tokens": result["input_tokens"] + result["output_tokens"],
                "cost": result["cost"],
                "duration": duration,
            })
        else:
            print_result(False, result["message"], {
                "tokens": result["input_tokens"] + result["output_tokens"],
                "cost": result["cost"],
                "duration": duration,
            })

        # Track in memory
        memory.track_model_performance(
            model=model,
            task_type="orchestrated",
            success=result["success"],
            tokens_used=result["input_tokens"] + result["output_tokens"],
            cost_usd=result["cost"],
            duration_ms=int(duration * 1000),
        )

        # Save to history
        history.add_task(
            task=task,
            result=result["message"],
            success=result["success"],
            model=model,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=result["cost"],
            subtasks=[{
                "id": st["id"],
                "task": st["task"],
                "status": st["status"],
                "retries": st.get("retries", 0),
            } for st in result.get("subtasks", [])],
        )


def run_single_grunt(
    task: str,
    model: str,
    context: str,
    working_dir: str,
    stream: bool,
    skip_qa: bool,
) -> dict:
    """Run a single grunt agent."""
    grunt = GruntAgent(model=model, working_dir=working_dir)

    if stream:
        # Streaming mode with live display
        display = StreamingDisplay(title=f"Grunt ({model})")

        def handle_stream(event: StreamEvent):
            display.handle_event(event)

        grunt.on_stream = handle_stream

        with display:
            result = grunt.run(task, context)
    else:
        # Non-streaming with spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Executing...", total=None)
            result = grunt.run(task, context)

    return {
        "success": result.success,
        "message": result.result,
        "error": result.error,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost": grunt.get_cost(),
        "files_modified": result.files_modified,
        "iterations": result.iterations,
    }


def run_orchestrated(
    task: str,
    model: str,
    context: str,
    working_dir: str,
    stream: bool,
    skip_qa: bool,
    parallel: bool,
) -> dict:
    """Run orchestrated multi-agent execution."""
    orchestrator = Orchestrator(model=model, working_dir=working_dir)

    # Set up logging callback
    log_messages = []

    def on_log(msg: str):
        log_messages.append(msg)
        if not stream:
            console.print(f"[dim]{msg}[/dim]")

    orchestrator.on_log = on_log

    if stream:
        # With streaming/live dashboard
        dashboard = LiveDashboard(task=task, model=model)

        def on_subtask_start(st):
            dashboard.add_grunt(st.id, st.task)
            dashboard.update_grunt(st.id, status="running")

        def on_subtask_complete(st):
            dashboard.update_grunt(
                st.id,
                status="completed" if st.status == "completed" else "failed",
                result=st.result.result[:100] if st.result else "",
            )

        def on_subtask_update(st):
            dashboard.update_grunt(st.id, retries=st.retries)

        orchestrator.on_subtask_start = on_subtask_start
        orchestrator.on_subtask_complete = on_subtask_complete
        orchestrator.on_subtask_update = on_subtask_update
        orchestrator.on_log = dashboard.log

        with dashboard:
            result = orchestrator.run(
                task,
                context=context,
                skip_qa=skip_qa,
                parallel=parallel,
                stream=True,
            )
    else:
        # Non-streaming with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            main_task = progress.add_task("Analyzing task...", total=None)

            def on_subtask_start(st):
                progress.update(main_task, description=f"Running subtask {st.id}...")

            def on_subtask_complete(st):
                status = "✓" if st.status == "completed" else "✗"
                progress.update(main_task, description=f"Subtask {st.id} {status}")

            orchestrator.on_subtask_start = on_subtask_start
            orchestrator.on_subtask_complete = on_subtask_complete

            result = orchestrator.run(
                task,
                context=context,
                skip_qa=skip_qa,
                parallel=parallel,
            )

    # Calculate cost
    costs = orchestrator.settings.cost_per_million.get(model, {"input": 0, "output": 0})
    total_cost = (
        (result.total_input_tokens / 1_000_000) * costs["input"] +
        (result.total_output_tokens / 1_000_000) * costs["output"]
    )

    return {
        "success": result.success,
        "message": result.message,
        "strategy": result.strategy,
        "input_tokens": result.total_input_tokens,
        "output_tokens": result.total_output_tokens,
        "cost": total_cost,
        "subtasks": [
            {
                "id": st.id,
                "task": st.task,
                "status": st.status,
                "retries": st.retries,
            }
            for st in result.subtasks
        ],
    }


@main.command()
@click.option("--limit", "-n", default=10, help="Number of sessions to show")
def history(limit: int):
    """List previous sessions."""
    sessions = ConversationHistory.list_sessions()

    if not sessions:
        console.print("[dim]No previous sessions found.[/dim]")
        return

    table = Table(title="Previous Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Tasks", justify="right")

    for session in sessions[:limit]:
        table.add_row(
            session["session_id"],
            session["created_at"][:19] if session["created_at"] else "-",
            str(session["task_count"]),
        )

    console.print(table)
    console.print("\n[dim]Resume a session with: swarm --session <session_id>[/dim]")


@main.command()
def stats():
    """Show usage statistics."""
    metrics = get_metrics_collector()
    metrics.load()

    if not metrics.tasks:
        console.print("[dim]No usage data found.[/dim]")
        return

    totals = metrics.get_totals()

    console.print(Panel(
        f"Total tasks: {totals['total_tasks']}\n"
        f"Total tokens: {format_tokens(totals['total_input_tokens'])} in / {format_tokens(totals['total_output_tokens'])} out\n"
        f"Total cost: {format_cost(totals['total_cost_usd'])}\n"
        f"Total retries: {totals['total_retries']}\n"
        f"Total time: {format_duration(totals['total_duration_seconds'])}",
        title="Usage Statistics",
        border_style=COLORS["primary"],
    ))


@main.command()
def memory():
    """Show memory statistics."""
    mem = get_memory_store()
    stats = mem.get_stats()
    model_stats = mem.get_model_stats()

    console.print(Panel(
        f"Total memories: {stats['total_memories']}\n"
        f"Successful: {stats['successful_memories']}\n"
        f"Success rate: {stats['success_rate']:.0%}\n"
        f"Total cost tracked: {format_cost(stats['total_cost_usd'])}\n"
        f"Skills learned: {stats['total_skills']}",
        title="Memory System",
        border_style=COLORS["secondary"],
    ))

    if model_stats:
        console.print()
        table = Table(title="Model Performance Tracking")
        table.add_column("Model")
        table.add_column("Tasks", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("Total Cost", justify="right")

        for model_name, data in sorted(model_stats.items()):
            table.add_row(
                model_name,
                str(data["total_tasks"]),
                f"{data['success_rate']:.0%}",
                format_tokens(int(data["avg_tokens"] or 0)),
                format_cost(data["total_cost"] or 0),
            )
        console.print(table)


if __name__ == "__main__":
    main()
