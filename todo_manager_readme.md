# TodoManager

A comprehensive task management system with JSON file persistence built on top of the existing Task model.

## Features

- **Add tasks** with title, description, priority, tags, and metadata
- **List tasks** with filtering by status, priority, or tags
- **Complete, delete, and update** existing tasks
- **Search tasks** by title or description content
- **JSON file persistence** - automatically saves and loads from file
- **Statistics** - get completion rates and task counts
- **Export/Import** - backup and restore tasks from JSON files
- **Comprehensive filtering** and sorting capabilities

## Quick Start

```python
from todo_manager import TodoManager
from task import TaskPriority, TaskStatus

# Initialize with a JSON file for persistence
manager = TodoManager("my_todos.json")

# Add a new task
task_id = manager.add_task(
    title="Complete project",
    description="Finish the final implementation",
    priority=TaskPriority.HIGH,
    tags=["work", "deadline"]
)

# List all tasks (sorted by priority)
all_tasks = manager.list_tasks()

# Filter by priority
high_priority_tasks = manager.list_tasks(priority_filter=TaskPriority.HIGH)

# Filter by status
pending_tasks = manager.list_tasks(status_filter=TaskStatus.PENDING)

# Filter by tag
work_tasks = manager.list_tasks(tag_filter="work")

# Search tasks
bug_tasks = manager.search_tasks("bug")

# Complete a task
manager.complete_task(task_id)

# Update a task
manager.update_task(task_id, title="New title", priority=TaskPriority.LOW)

# Get statistics
stats = manager.get_stats()
print(f"Completion rate: {stats['completion_rate']}%")

# Export for backup
manager.export_to_json("backup.json")
```

## API Reference

### TodoManager(file_path="todos.json")
Initialize TodoManager with JSON file for persistence.

### Core Methods

- **add_task(title, description=None, priority=MEDIUM, tags=None, metadata=None)** → str
  - Add a new task and return its ID

- **list_tasks(status_filter=None, priority_filter=None, tag_filter=None)** → List[Task]
  - List tasks with optional filtering and priority sorting

- **get_task(task_id)** → Optional[Task]
  - Get a specific task by ID

- **complete_task(task_id)** → bool
  - Mark task as completed

- **delete_task(task_id)** → bool
  - Remove task from manager

- **update_task(task_id, **kwargs)** → bool
  - Update any task fields

- **search_tasks(query)** → List[Task]
  - Search tasks by title/description content

### Utility Methods

- **get_stats()** → Dict
  - Get task statistics and completion rate

- **clear_completed_tasks()** → int
  - Remove all completed tasks

- **export_to_json(path)** → None
  - Export tasks to JSON file

- **import_from_json(path, merge=False)** → int
  - Import tasks from JSON file

## Task Model

The TodoManager uses the existing Task dataclass with:

- **id**: Unique identifier (UUID)
- **title**: Task title (required)
- **description**: Optional detailed description
- **priority**: TaskPriority enum (LOW, MEDIUM, HIGH, CRITICAL)
- **status**: TaskStatus enum (PENDING, IN_PROGRESS, COMPLETED, BLOCKED, CANCELLED)
- **created_at**: Creation timestamp
- **updated_at**: Last update timestamp
- **tags**: List of string tags
- **metadata**: Dictionary for additional data

## File Format

Tasks are persisted as JSON with the following structure:

```json
{
  "task-uuid": {
    "id": "task-uuid",
    "title": "Task Title",
    "description": "Task description",
    "priority": "HIGH",
    "status": "PENDING",
    "created_at": "2025-12-23T20:41:34.305920",
    "updated_at": null,
    "tags": ["work", "urgent"],
    "metadata": {"project": "alpha"}
  }
}
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest test_todo_manager.py -v
```

## Examples

See `todo_demo.py` for a complete demonstration of all features.

## Error Handling

- File I/O errors are caught and reported
- Invalid task IDs return False/None gracefully
- JSON parsing errors show warnings but don't crash
- Missing files are created automatically