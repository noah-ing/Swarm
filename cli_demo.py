#!/usr/bin/env python3
"""
Todo CLI Demo - Interactive demonstration of the command-line interface.

This script demonstrates all the features of the TodoCLI including:
- Adding tasks with different priorities and tags
- Listing and filtering tasks
- Completing and deleting tasks  
- Searching and updating tasks
- Statistics and management operations
- Export/import functionality
"""

import tempfile
import os
import json
from pathlib import Path
from todo_cli import TodoCLI


def demo_header(title: str) -> None:
    """Print a formatted demo section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def demo_command(cli: TodoCLI, args: list, description: str) -> None:
    """Execute a CLI command with description."""
    print(f"\n► {description}")
    print(f"  Command: todo {' '.join(args)}")
    print(f"  Output:")
    try:
        cli.run(args)
    except SystemExit:
        pass  # Normal for some commands like help


def main():
    """Run the interactive CLI demo."""
    print("Todo CLI - Interactive Demo")
    print("This demo showcases all CLI features with sample data")
    
    # Create temporary file for demo
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'demo_todos.json')
    
    try:
        cli = TodoCLI()
        
        # Base args with temp file
        base_args = ['-f', temp_file]
        
        demo_header("BASIC OPERATIONS")
        
        # Add tasks
        demo_command(cli, base_args + ['add', 'Buy groceries', '-p', 'high', '-t', 'shopping,food', '-d', 'Get milk, bread, and eggs'],
                    "Add a high-priority grocery task with tags and description")
        
        demo_command(cli, base_args + ['add', 'Walk the dog', '-p', 'medium', '-t', 'pets,exercise'],
                    "Add a medium-priority exercise task")
        
        demo_command(cli, base_args + ['add', 'Complete project report', '-p', 'critical', '-t', 'work,deadline'],
                    "Add a critical work task")
        
        demo_command(cli, base_args + ['add', 'Read a book', '-p', 'low', '-t', 'leisure,personal'],
                    "Add a low-priority leisure task")
        
        demo_command(cli, base_args + ['add', 'Fix kitchen sink', '-t', 'home,maintenance'],
                    "Add a home maintenance task (default medium priority)")
        
        # List all tasks
        demo_command(cli, base_args + ['list'],
                    "List all tasks (sorted by priority)")
        
        demo_header("FILTERING AND SEARCHING")
        
        # List with filters
        demo_command(cli, base_args + ['list', '-p', 'high'],
                    "List only high-priority tasks")
        
        demo_command(cli, base_args + ['list', '-t', 'work'],
                    "List tasks with 'work' tag")
        
        demo_command(cli, base_args + ['list', '--format', 'brief'],
                    "List tasks in brief format")
        
        demo_command(cli, base_args + ['list', '--format', 'json', '--limit', '2'],
                    "List first 2 tasks in JSON format")
        
        # Search tasks
        demo_command(cli, base_args + ['search', 'project'],
                    "Search for tasks containing 'project'")
        
        demo_command(cli, base_args + ['search', 'dog', '--format', 'brief'],
                    "Search for tasks containing 'dog' in brief format")
        
        demo_header("TASK MANAGEMENT")
        
        # Get task IDs for operations
        cli.init_manager(temp_file)
        tasks = cli.manager.list_tasks()
        
        if tasks:
            # Show task details
            demo_command(cli, base_args + ['show', tasks[0].id],
                        f"Show detailed information for task: {tasks[0].title}")
            
            # Complete a task
            demo_command(cli, base_args + ['complete', tasks[1].id],
                        f"Complete task: {tasks[1].title}")
            
            # Update a task
            demo_command(cli, base_args + ['update', tasks[2].id, '--priority', 'high', '--status', 'in_progress'],
                        f"Update task priority and status: {tasks[2].title}")
            
            # List after changes
            demo_command(cli, base_args + ['list'],
                        "List all tasks after completing and updating")
        
        demo_header("STATISTICS AND ANALYSIS")
        
        # Show statistics
        demo_command(cli, base_args + ['stats'],
                    "Show task statistics and summary")
        
        # List completed tasks
        demo_command(cli, base_args + ['list', '-s', 'completed'],
                    "List completed tasks")
        
        # List pending tasks
        demo_command(cli, base_args + ['list', '-s', 'pending'],
                    "List pending tasks")
        
        demo_header("IMPORT/EXPORT")
        
        # Export tasks
        export_file = os.path.join(temp_dir, 'exported_tasks.json')
        demo_command(cli, base_args + ['export', export_file],
                    "Export all tasks to JSON file")
        
        # Show exported content
        print(f"\n► Exported file content (first 500 chars):")
        with open(export_file, 'r') as f:
            content = f.read()
            print(f"  {content[:500]}{'...' if len(content) > 500 else ''}")
        
        # Create new todo file and import
        import_file = os.path.join(temp_dir, 'imported_todos.json')
        demo_command(cli, ['-f', import_file, 'import', export_file],
                    "Import tasks to new todo file")
        
        demo_command(cli, ['-f', import_file, 'list'],
                    "List tasks in imported file")
        
        demo_header("BULK OPERATIONS")
        
        if len(tasks) >= 2:
            # Complete multiple tasks
            task_ids = [t.id for t in tasks[:2]]
            demo_command(cli, base_args + ['complete'] + task_ids,
                        "Complete multiple tasks at once")
            
            # Clear completed tasks
            demo_command(cli, base_args + ['clear', '--force'],
                        "Clear all completed tasks")
        
        demo_header("ERROR HANDLING EXAMPLES")
        
        # Try to show non-existent task
        print(f"\n► Try to show non-existent task (demonstrates error handling)")
        print(f"  Command: todo {' '.join(base_args + ['show', 'non-existent-id'])}")
        print(f"  Output:")
        try:
            cli.run(base_args + ['show', 'non-existent-id'])
        except SystemExit as e:
            print(f"    Error handled gracefully (exit code: {e.code})")
        
        # Try invalid priority
        print(f"\n► Try to add task with invalid priority (demonstrates validation)")
        print(f"  Command: todo {' '.join(base_args + ['add', 'Test', '-p', 'invalid'])}")
        print(f"  Output:")
        try:
            cli.run(base_args + ['add', 'Test', '-p', 'invalid'])
        except SystemExit as e:
            print(f"    Validation error handled (exit code: {e.code})")
        
        demo_header("HELP AND USAGE")
        
        # Show help
        demo_command(cli, ['--help'],
                    "Show main help")
        
        demo_command(cli, ['add', '--help'],
                    "Show help for add command")
        
        demo_header("FINAL STATISTICS")
        
        # Final stats
        demo_command(cli, base_args + ['stats'],
                    "Final task statistics")
        
        demo_command(cli, base_args + ['list'],
                    "Final task list")
        
        print(f"\n{'='*60}")
        print("  DEMO COMPLETE")
        print("  All CLI features demonstrated successfully!")
        print("  Temporary files created in:", temp_dir)
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Demo error: {e}")
    
    finally:
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"\nTemporary files cleaned up.")
        except:
            pass


if __name__ == '__main__':
    main()