#!/usr/bin/env python3
"""
Todo CLI - Command line interface for managing tasks

Provides simple command-line operations:
- add: Add a new task
- list: List all tasks
- complete: Mark a task as completed
- delete: Delete a task
"""

import argparse
import sys
from pathlib import Path

from todo_manager import TodoManager
from task import TaskPriority, TaskStatus


def add_task(manager: TodoManager, args: argparse.Namespace) -> None:
    """Add a new task."""
    try:
        # Parse priority if provided
        priority = TaskPriority.MEDIUM
        if args.priority:
            priority_map = {
                'low': TaskPriority.LOW,
                'medium': TaskPriority.MEDIUM,
                'high': TaskPriority.HIGH,
                'critical': TaskPriority.CRITICAL
            }
            priority = priority_map.get(args.priority.lower(), TaskPriority.MEDIUM)
        
        # Add the task
        task_id = manager.add_task(
            title=args.title,
            description=args.description,
            priority=priority
        )
        
        print(f"✓ Task added successfully (ID: {task_id[:8]}...)")
        
    except Exception as e:
        print(f"✗ Error adding task: {e}", file=sys.stderr)
        sys.exit(1)


def list_tasks(manager: TodoManager, args: argparse.Namespace) -> None:
    """List all tasks."""
    try:
        # Get all tasks
        tasks = manager.list_tasks()
        
        if not tasks:
            print("No tasks found.")
            return
        
        # Print header
        print(f"\n{'ID':^10} | {'Title':^30} | {'Priority':^10} | {'Status':^12}")
        print("-" * 70)
        
        # Print tasks
        for task in tasks:
            task_id_short = task.id[:8] if task.id else 'N/A'
            title_short = task.title[:30] if len(task.title) > 30 else task.title
            print(f"{task_id_short:^10} | {title_short:<30} | {task.priority.name:^10} | {task.status.name:^12}")
        
        print(f"\nTotal tasks: {len(tasks)}")
        
    except Exception as e:
        print(f"✗ Error listing tasks: {e}", file=sys.stderr)
        sys.exit(1)


def complete_task(manager: TodoManager, args: argparse.Namespace) -> None:
    """Mark a task as completed."""
    try:
        # Find task by ID prefix
        task_id = find_task_by_prefix(manager, args.task_id)
        
        if not task_id:
            print(f"✗ No task found with ID starting with: {args.task_id}", file=sys.stderr)
            sys.exit(1)
        
        # Complete the task
        if manager.complete_task(task_id):
            task = manager.get_task(task_id)
            print(f"✓ Task completed: {task.title}")
        else:
            print(f"✗ Failed to complete task", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error completing task: {e}", file=sys.stderr)
        sys.exit(1)


def delete_task(manager: TodoManager, args: argparse.Namespace) -> None:
    """Delete a task."""
    try:
        # Find task by ID prefix
        task_id = find_task_by_prefix(manager, args.task_id)
        
        if not task_id:
            print(f"✗ No task found with ID starting with: {args.task_id}", file=sys.stderr)
            sys.exit(1)
        
        # Get task details before deletion
        task = manager.get_task(task_id)
        task_title = task.title if task else "Unknown"
        
        # Confirm deletion if not forced
        if not args.force:
            response = input(f"Are you sure you want to delete '{task_title}'? (y/N): ")
            if response.lower() != 'y':
                print("Deletion cancelled.")
                return
        
        # Delete the task
        if manager.delete_task(task_id):
            print(f"✓ Task deleted: {task_title}")
        else:
            print(f"✗ Failed to delete task", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error deleting task: {e}", file=sys.stderr)
        sys.exit(1)


def find_task_by_prefix(manager: TodoManager, prefix: str) -> str:
    """Find a task ID by its prefix."""
    matching_ids = [tid for tid in manager.tasks.keys() if tid.startswith(prefix)]
    
    if len(matching_ids) == 1:
        return matching_ids[0]
    elif len(matching_ids) > 1:
        print(f"✗ Multiple tasks found with prefix '{prefix}':", file=sys.stderr)
        for tid in matching_ids[:5]:  # Show first 5 matches
            task = manager.get_task(tid)
            print(f"  - {tid[:8]}: {task.title}", file=sys.stderr)
        if len(matching_ids) > 5:
            print(f"  ... and {len(matching_ids) - 5} more", file=sys.stderr)
        sys.exit(1)
    
    return None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simple todo task manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s add "Buy groceries" -p high
  %(prog)s list
  %(prog)s complete abc123
  %(prog)s delete abc123 --force
        """
    )
    
    # Global options
    parser.add_argument(
        '-f', '--file',
        default='todos.json',
        help='Path to the todo JSON file (default: todos.json)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new task')
    add_parser.add_argument('title', help='Task title')
    add_parser.add_argument('-d', '--description', help='Task description')
    add_parser.add_argument(
        '-p', '--priority',
        choices=['low', 'medium', 'high', 'critical'],
        default='medium',
        help='Task priority (default: medium)'
    )
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all tasks')
    
    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Mark a task as completed')
    complete_parser.add_argument('task_id', help='Task ID (or prefix)')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a task')
    delete_parser.add_argument('task_id', help='Task ID (or prefix)')
    delete_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Delete without confirmation'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Initialize todo manager
        manager = TodoManager(args.file)
        
        # Execute command
        if args.command == 'add':
            add_task(manager, args)
        elif args.command == 'list':
            list_tasks(manager, args)
        elif args.command == 'complete':
            complete_task(manager, args)
        elif args.command == 'delete':
            delete_task(manager, args)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()