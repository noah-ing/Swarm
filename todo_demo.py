#!/usr/bin/env python3
"""
Demo script for TodoManager functionality.

This script demonstrates how to use the TodoManager class with all its features.
"""

from todo_manager import TodoManager
from task import TaskPriority, TaskStatus


def main():
    """Demonstrate TodoManager functionality"""
    print("=== TodoManager Demo ===\n")
    
    # Initialize TodoManager with a demo file
    print("1. Initializing TodoManager...")
    manager = TodoManager("demo_todos.json")
    print(f"   TodoManager initialized with file: {manager.file_path}")
    
    # Add some tasks
    print("\n2. Adding tasks...")
    
    task1_id = manager.add_task(
        title="Complete project documentation",
        description="Write comprehensive documentation for the new API",
        priority=TaskPriority.HIGH,
        tags=["work", "documentation", "api"]
    )
    print(f"   Added task: {manager.get_task(task1_id).title}")
    
    task2_id = manager.add_task(
        title="Buy groceries",
        description="Milk, bread, eggs, and vegetables",
        priority=TaskPriority.MEDIUM,
        tags=["personal", "shopping"]
    )
    print(f"   Added task: {manager.get_task(task2_id).title}")
    
    task3_id = manager.add_task(
        title="Fix critical security bug",
        description="Address the authentication vulnerability in user login",
        priority=TaskPriority.CRITICAL,
        tags=["work", "security", "urgent"]
    )
    print(f"   Added task: {manager.get_task(task3_id).title}")
    
    task4_id = manager.add_task(
        title="Plan weekend trip",
        priority=TaskPriority.LOW,
        tags=["personal", "travel"]
    )
    print(f"   Added task: {manager.get_task(task4_id).title}")
    
    # List all tasks
    print("\n3. Listing all tasks (sorted by priority):")
    all_tasks = manager.list_tasks()
    for i, task in enumerate(all_tasks, 1):
        print(f"   {i}. [{task.priority.name}] {task.title}")
        if task.description:
            print(f"      {task.description}")
        print(f"      Tags: {', '.join(task.tags) if task.tags else 'None'}")
        print(f"      Status: {task.status.name}")
        print()
    
    # Filter by priority
    print("4. High priority tasks only:")
    high_priority_tasks = manager.list_tasks(priority_filter=TaskPriority.HIGH)
    for task in high_priority_tasks:
        print(f"   - {task.title}")
    
    print("\n5. Critical priority tasks only:")
    critical_tasks = manager.list_tasks(priority_filter=TaskPriority.CRITICAL)
    for task in critical_tasks:
        print(f"   - {task.title}")
    
    # Filter by tags
    print("\n6. Work-related tasks:")
    work_tasks = manager.list_tasks(tag_filter="work")
    for task in work_tasks:
        print(f"   - {task.title}")
    
    # Search functionality
    print("\n7. Searching for tasks containing 'bug':")
    bug_tasks = manager.search_tasks("bug")
    for task in bug_tasks:
        print(f"   - {task.title}")
    
    # Complete some tasks
    print("\n8. Completing tasks...")
    print(f"   Completing: {manager.get_task(task2_id).title}")
    manager.complete_task(task2_id)
    
    print(f"   Updating status of: {manager.get_task(task3_id).title}")
    manager.update_task(task3_id, status=TaskStatus.IN_PROGRESS)
    
    # Show stats
    print("\n9. Task statistics:")
    stats = manager.get_stats()
    print(f"   Total tasks: {stats['total']}")
    print(f"   Completion rate: {stats['completion_rate']}%")
    print("   By status:")
    for status, count in stats['by_status'].items():
        print(f"     - {status}: {count}")
    print("   By priority:")
    for priority, count in stats['by_priority'].items():
        print(f"     - {priority}: {count}")
    
    # List pending tasks only
    print("\n10. Pending tasks:")
    pending_tasks = manager.list_tasks(status_filter=TaskStatus.PENDING)
    for task in pending_tasks:
        print(f"    - {task.title}")
    
    # Update a task
    print("\n11. Updating a task...")
    old_title = manager.get_task(task4_id).title
    manager.update_task(
        task4_id, 
        title="Plan summer vacation to Europe",
        description="Research destinations, book flights, and accommodations",
        priority=TaskPriority.MEDIUM,
        tags=["personal", "travel", "europe", "vacation"]
    )
    print(f"    Updated: '{old_title}' → '{manager.get_task(task4_id).title}'")
    
    # Export tasks
    print("\n12. Exporting tasks to backup file...")
    manager.export_to_json("backup_todos.json")
    print("    Tasks exported to backup_todos.json")
    
    # Final task overview
    print("\n13. Final task overview:")
    final_tasks = manager.list_tasks()
    print(f"    Total: {len(final_tasks)} tasks")
    for status in TaskStatus:
        count = len(manager.list_tasks(status_filter=status))
        if count > 0:
            print(f"    {status.name}: {count}")
    
    print("\n=== Demo Complete ===")
    print(f"Tasks have been saved to: {manager.file_path}")
    print("You can examine the JSON file to see the persisted data.")


if __name__ == "__main__":
    main()