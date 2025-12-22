#!/usr/bin/env python3
"""Swarm: A multi-agent CLI for autonomous task execution."""

import os
import sys
import uuid

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from config import get_settings
from agents import GruntAgent, Orchestrator
from router import ModelRouter
from history import ConversationHistory, get_history
from logging_config import setup_logging, get_metrics_collector, TaskMetrics
from context import ContextBuilder


console = Console()


def format_cost(cost: float) -> str:
    """Format cost in dollars."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def format_tokens(tokens: int) -> str:
    """Format token count."""
    if tokens > 1000:
        return f"{tokens / 1000:.1f}K"
    return str(tokens)


@click.group(invoke_without_command=True)
@click.argument("task", required=False)
@click.option("--prefer", "-p", type=click.Choice(["anthropic", "openai"]), help="Preferred provider")
@click.option("--model", "-m", help="Force a specific model (haiku, sonnet, opus, gpt-4o-mini, gpt-4o)")
@click.option("--cheap", is_flag=True, help="Use cheapest models (haiku/gpt-4o-mini)")
@click.option("--single", "-s", is_flag=True, help="Single grunt mode (no orchestration)")
@click.option("--no-qa", is_flag=True, help="Skip QA validation")
@click.option("--no-parallel", is_flag=True, help="Disable parallel execution")
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
        swarm -i  # Interactive mode

    \b
    Commands:
        swarm history    List previous sessions
        swarm stats      Show usage statistics
    """
    # If a subcommand is invoked, don't run main logic
    if ctx.invoked_subcommand is not None:
        return

    settings = get_settings()

    # Set up logging
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

    # Resolve working directory
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
    verbose: bool,
    working_dir: str,
    history: ConversationHistory,
):
    """Run in interactive mode."""
    console.print(Panel.fit(
        "[bold blue]Swarm[/bold blue] - Multi-Agent Task Executor\n"
        f"Session: {history.session_id}\n"
        "Type your task, or 'quit' to exit.\n"
        "Commands: /history, /stats, /clear",
        border_style="blue"
    ))

    while True:
        try:
            task = console.input("\n[bold green]swarm>[/bold green] ").strip()
            if not task:
                continue

            # Handle commands
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
                verbose=verbose,
                working_dir=working_dir,
                history=history,
            )

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
            status = "[green]✓[/green]" if task.success else "[red]✗[/red]"
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
            border_style="blue",
        ))

    elif cmd == "/clear":
        console.clear()
        console.print("[dim]Screen cleared.[/dim]")

    elif cmd == "/help":
        console.print(Panel(
            "/history - Show task history\n"
            "/stats   - Show usage statistics\n"
            "/clear   - Clear screen\n"
            "/help    - Show this help\n"
            "quit     - Exit and save session",
            title="Commands",
            border_style="dim",
        ))

    else:
        console.print(f"[yellow]Unknown command: {command}[/yellow]")
        console.print("[dim]Type /help for available commands.[/dim]")


def run_task(
    task: str,
    prefer: str | None,
    model: str | None,
    cheap: bool,
    single: bool,
    no_qa: bool,
    no_parallel: bool,
    verbose: bool,
    working_dir: str,
    history: ConversationHistory,
):
    """Execute a single task."""
    settings = get_settings()
    metrics = get_metrics_collector()
    task_metrics = metrics.start_task(history.generate_task_id())

    # Determine model
    if cheap:
        model = "haiku" if (prefer or settings.prefer_provider) == "anthropic" else "gpt-4o-mini"
    elif not model:
        router = ModelRouter(prefer_provider=prefer)
        model = router.select(task)

    console.print(f"\n[dim]Model: {model} | Working dir: {working_dir}[/dim]")

    # Add history context
    history_context = history.get_context_summary()

    if single:
        # Single grunt mode
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Executing task...", total=None)

            grunt = GruntAgent(model=model, working_dir=working_dir)
            result = grunt.run(task, context=history_context)

        if result.success:
            console.print(Panel(
                Markdown(result.result),
                title="[green]Task Completed[/green]",
                border_style="green",
            ))

            if result.files_modified:
                console.print(f"\n[dim]Files modified: {', '.join(result.files_modified)}[/dim]")

        else:
            console.print(Panel(
                result.error or "Unknown error",
                title="[red]Task Failed[/red]",
                border_style="red",
            ))

        # Record metrics
        cost = grunt.get_cost()
        task_metrics.input_tokens = result.input_tokens
        task_metrics.output_tokens = result.output_tokens
        task_metrics.cost_usd = cost
        task_metrics.complete()

        # Save to history
        history.add_task(
            task=task,
            result=result.result if result.success else (result.error or "Failed"),
            success=result.success,
            model=model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=cost,
            files_modified=result.files_modified,
        )

        # Stats
        console.print(f"\n[dim]Tokens: {format_tokens(result.input_tokens)} in / {format_tokens(result.output_tokens)} out | Cost: {format_cost(cost)} | Iterations: {result.iterations}[/dim]")

    else:
        # Orchestrated mode
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            main_task = progress.add_task("Decomposing task...", total=None)

            orchestrator = Orchestrator(model=model, working_dir=working_dir)

            # Set up progress callbacks
            subtask_progress = {}

            def on_subtask_start(subtask):
                subtask_progress[subtask.id] = progress.add_task(
                    f"  [{subtask.id}] {subtask.task[:40]}...",
                    total=None,
                )

            def on_subtask_complete(subtask):
                if subtask.id in subtask_progress:
                    progress.update(subtask_progress[subtask.id], completed=True)

            orchestrator.on_subtask_start = on_subtask_start
            orchestrator.on_subtask_complete = on_subtask_complete

            result = orchestrator.run(
                task,
                context=history_context,
                skip_qa=no_qa,
                parallel=not no_parallel,
            )

            progress.update(main_task, description="Complete", completed=True)

        # Show subtask results
        if result.subtasks:
            table = Table(title="Subtasks", show_header=True)
            table.add_column("#", style="dim", width=3)
            table.add_column("Task", max_width=50)
            table.add_column("Status", width=10)
            table.add_column("Retries", justify="right", width=7)

            for st in result.subtasks:
                status_style = {
                    "completed": "[green]✓ done[/green]",
                    "failed": "[red]✗ failed[/red]",
                    "pending": "[yellow]○ pending[/yellow]",
                    "running": "[blue]◉ running[/blue]",
                }.get(st.status, st.status)

                table.add_row(
                    str(st.id),
                    st.task[:50] + ("..." if len(st.task) > 50 else ""),
                    status_style,
                    str(st.retries) if st.retries > 0 else "-",
                )

            console.print(table)

        # Final result
        if result.success:
            console.print(Panel(
                Markdown(result.message),
                title="[green]All Tasks Completed[/green]",
                border_style="green",
            ))
        else:
            console.print(Panel(
                result.message,
                title="[red]Task Failed[/red]",
                border_style="red",
            ))

            # Show failed subtask details
            for st in result.subtasks:
                if st.status == "failed" and st.result:
                    console.print(f"\n[red]Subtask {st.id} failed:[/red] {st.result.error or 'Unknown error'}")

        # Calculate cost
        costs = settings.cost_per_million.get(model, {"input": 0, "output": 0})
        total_cost = (
            (result.total_input_tokens / 1_000_000) * costs["input"] +
            (result.total_output_tokens / 1_000_000) * costs["output"]
        )

        # Record metrics
        task_metrics.input_tokens = result.total_input_tokens
        task_metrics.output_tokens = result.total_output_tokens
        task_metrics.cost_usd = total_cost
        task_metrics.subtasks_total = len(result.subtasks)
        task_metrics.subtasks_completed = sum(1 for st in result.subtasks if st.status == "completed")
        task_metrics.subtasks_failed = sum(1 for st in result.subtasks if st.status == "failed")
        task_metrics.retries = sum(st.retries for st in result.subtasks)
        task_metrics.complete()

        # Save to history
        history.add_task(
            task=task,
            result=result.message,
            success=result.success,
            model=model,
            input_tokens=result.total_input_tokens,
            output_tokens=result.total_output_tokens,
            cost_usd=total_cost,
            subtasks=[{
                "id": st.id,
                "task": st.task,
                "status": st.status,
                "retries": st.retries,
            } for st in result.subtasks],
        )

        # Stats
        console.print(f"\n[dim]Tokens: {format_tokens(result.total_input_tokens)} in / {format_tokens(result.total_output_tokens)} out | Cost: {format_cost(total_cost)} | Subtasks: {len(result.subtasks)}[/dim]")


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
        f"Total time: {totals['total_duration_seconds']:.1f}s",
        title="Usage Statistics",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
