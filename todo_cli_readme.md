# Todo CLI - Command Line Interface

A comprehensive command-line interface for todo task management with full CRUD operations, filtering, searching, and import/export functionality.

## Features

### Core Operations
- **Add tasks** with title, description, priority, tags, and metadata
- **List tasks** with filtering by status, priority, and tags
- **Complete tasks** individually or in bulk
- **Delete tasks** with confirmation prompts
- **Update tasks** with new properties
- **Search tasks** by title or description content
- **Show detailed** task information

### Advanced Features
- **Statistics and analytics** with completion rates and breakdowns
- **Import/Export** tasks from/to JSON files with merge options
- **Multiple output formats** (table, JSON, brief)
- **Bulk operations** for efficient task management
- **Error handling** with user-friendly messages
- **Confirmation prompts** for destructive operations

## Installation

The CLI requires the existing TodoManager and Task classes:

```bash
# Ensure you have the required dependencies
pip install pytest  # for running tests

# Make the CLI executable
chmod +x todo_cli.py
```

## Basic Usage

### Adding Tasks

```bash
# Basic task
python todo_cli.py add "Buy groceries"

# Task with priority and tags
python todo_cli.py add "Complete project" -p high -t work,deadline

# Task with full details
python todo_cli.py add "Plan vacation" \
  -p medium \
  -d "Research destinations and book flights" \
  -t travel,personal \
  -m '{"budget": 5000, "duration": "1 week"}'
```

### Listing Tasks

```bash
# List all tasks
python todo_cli.py list

# List with filters
python todo_cli.py list -s pending -p high
python todo_cli.py list -t work

# Different output formats
python todo_cli.py list --format brief
python todo_cli.py list --format json --limit 5
```

### Task Management

```bash
# Complete tasks
python todo_cli.py complete <task-id>
python todo_cli.py complete <id1> <id2> <id3>  # bulk complete

# Update task
python todo_cli.py update <task-id> --priority critical --status in_progress

# Show task details
python todo_cli.py show <task-id>

# Delete tasks (with confirmation)
python todo_cli.py delete <task-id>
python todo_cli.py delete <task-id> --force  # skip confirmation
```

### Searching and Statistics

```bash
# Search tasks
python todo_cli.py search "grocery"
python todo_cli.py search "project" --format brief

# View statistics
python todo_cli.py stats

# Clear completed tasks
python todo_cli.py clear
```

### Import/Export

```bash
# Export tasks
python todo_cli.py export backup.json

# Import tasks (replace existing)
python todo_cli.py import backup.json

# Import and merge with existing
python todo_cli.py import backup.json --merge
```

## Command Reference

### Global Options

| Option | Description |
|--------|-------------|
| `-f, --file FILE` | Path to JSON file (default: todos.json) |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Show help message |

### Commands

#### add
Add a new task to the todo list.

```
python todo_cli.py add TITLE [options]
```

**Options:**
- `-d, --description TEXT` - Detailed description
- `-p, --priority {low,medium,high,critical}` - Priority level (default: medium)
- `-t, --tags TAGS` - Comma-separated list of tags
- `-m, --metadata JSON` - JSON string of additional metadata

**Examples:**
```bash
python todo_cli.py add "Buy groceries" -p high -t shopping,food
python todo_cli.py add "Complete report" -d "Quarterly sales analysis" -p critical
```

#### list
List tasks with optional filtering and formatting.

```
python todo_cli.py list [options]
```

**Options:**
- `-s, --status {pending,in_progress,completed,blocked,cancelled}` - Filter by status
- `-p, --priority {low,medium,high,critical}` - Filter by priority
- `-t, --tag TAG` - Filter by tag (exact match)
- `--format {table,json,brief}` - Output format (default: table)
- `--limit N` - Maximum number of tasks to display

**Examples:**
```bash
python todo_cli.py list -s pending -p high
python todo_cli.py list --format json --limit 10
python todo_cli.py list -t work
```

#### complete
Mark one or more tasks as completed.

```
python todo_cli.py complete TASK_ID [TASK_ID ...]
```

**Examples:**
```bash
python todo_cli.py complete abc123def-456-789
python todo_cli.py complete id1 id2 id3  # bulk complete
```

#### delete
Delete one or more tasks.

```
python todo_cli.py delete TASK_ID [TASK_ID ...] [options]
```

**Options:**
- `--force` - Skip confirmation prompt

**Examples:**
```bash
python todo_cli.py delete abc123def-456-789
python todo_cli.py delete id1 id2 --force
```

#### update
Update an existing task's properties.

```
python todo_cli.py update TASK_ID [options]
```

**Options:**
- `--title TEXT` - New title
- `--description TEXT` - New description
- `--priority {low,medium,high,critical}` - New priority
- `--status {pending,in_progress,completed,blocked,cancelled}` - New status
- `--tags TAGS` - New comma-separated list of tags

**Examples:**
```bash
python todo_cli.py update abc123 --priority high --status in_progress
python todo_cli.py update abc123 --title "Updated title" --tags work,urgent
```

#### search
Search tasks by title or description.

```
python todo_cli.py search QUERY [options]
```

**Options:**
- `--format {table,json,brief}` - Output format (default: table)

**Examples:**
```bash
python todo_cli.py search "grocery"
python todo_cli.py search "project" --format brief
```

#### show
Display detailed information about a specific task.

```
python todo_cli.py show TASK_ID
```

**Examples:**
```bash
python todo_cli.py show abc123def-456-789
```

#### stats
Show task statistics and summary information.

```
python todo_cli.py stats
```

Shows:
- Total task count
- Completion rate percentage
- Breakdown by status
- Breakdown by priority

#### clear
Remove all completed tasks.

```
python todo_cli.py clear [options]
```

**Options:**
- `--force` - Skip confirmation prompt

**Examples:**
```bash
python todo_cli.py clear
python todo_cli.py clear --force
```

#### export
Export all tasks to a JSON file.

```
python todo_cli.py export OUTPUT_FILE
```

**Examples:**
```bash
python todo_cli.py export backup.json
python todo_cli.py export ~/Documents/todos_backup.json
```

#### import
Import tasks from a JSON file.

```
python todo_cli.py import INPUT_FILE [options]
```

**Options:**
- `--merge` - Merge with existing tasks instead of replacing

**Examples:**
```bash
python todo_cli.py import backup.json
python todo_cli.py import backup.json --merge
```

## Output Formats

### Table Format (default)
Tabular display with columns for ID, Title, Status, Priority, and Creation date.

```
ID       Title         Status   Priority Created
------------------------------------------------
abc12345 Buy groceries PENDING  HIGH     2023-12-01 10:30
```

### Brief Format
Compact single-line format with status indicators.

```
○ abc12345 !Buy groceries
✓ def67890 Walk the dog
```

Icons:
- `○` - Pending/In Progress
- `✓` - Completed
- `!` - High/Critical priority

### JSON Format
Full JSON representation of tasks.

```json
[
  {
    "id": "abc12345-def6-7890-abcd-ef1234567890",
    "title": "Buy groceries",
    "status": "PENDING",
    "priority": "HIGH",
    "created_at": "2023-12-01T10:30:00",
    "tags": ["shopping", "food"],
    "metadata": {}
  }
]
```

## Error Handling

The CLI provides comprehensive error handling:

### Input Validation
- Invalid priority levels
- Malformed JSON metadata
- Invalid status values
- Missing required arguments

### File Operations
- Non-existent files for import
- Permission errors for file access
- Invalid JSON format in import files

### Task Operations
- Non-existent task IDs
- Empty task lists
- Failed save operations

All errors display user-friendly messages and appropriate exit codes.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (task not found, operation failed) |
| 2 | Invalid arguments or validation error |

## Examples and Use Cases

### Daily Task Management
```bash
# Start the day - see what's pending
python todo_cli.py list -s pending

# Add today's tasks
python todo_cli.py add "Team meeting" -p high -t work
python todo_cli.py add "Grocery shopping" -t personal

# Complete tasks as you go
python todo_cli.py complete <task-id>

# End of day - check stats
python todo_cli.py stats
```

### Project Management
```bash
# Add project tasks with priorities
python todo_cli.py add "Design mockups" -p high -t project,design
python todo_cli.py add "Write documentation" -p medium -t project,docs
python todo_cli.py add "Code review" -p critical -t project,review

# Filter by project tag
python todo_cli.py list -t project

# Update priorities as needed
python todo_cli.py update <task-id> --priority critical
```

### Backup and Migration
```bash
# Backup current tasks
python todo_cli.py export backup_$(date +%Y%m%d).json

# Migrate to new file
python todo_cli.py -f new_todos.json import backup_20231201.json

# Merge tasks from multiple sources
python todo_cli.py import team_tasks.json --merge
```

## Testing

Run the comprehensive test suite:

```bash
# Run all CLI tests
python -m pytest test_todo_cli.py -v

# Run integration tests
python -m pytest test_todo_cli.py::TestTodoCLIIntegration -v

# Run demo
python cli_demo.py
```

## Architecture

The CLI is built with:

- **argparse** for robust argument parsing
- **Modular command design** for easy extension
- **Comprehensive error handling** with user-friendly messages
- **Multiple output formats** for different use cases
- **Confirmation prompts** for destructive operations
- **Extensive test coverage** (36 test cases)

The CLI acts as a thin interface layer over the TodoManager class, providing command-line access to all core functionality while adding CLI-specific features like formatting, confirmation prompts, and error handling.