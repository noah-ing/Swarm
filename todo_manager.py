import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from task import Task, TaskPriority, TaskStatus


class TodoManager:
    """
    TodoManager class for managing tasks with JSON file persistence.
    
    Provides methods to add, list, complete, delete tasks and handle
    JSON file persistence for data storage.
    """
    
    def __init__(self, file_path: str = "todos.json"):
        """
        Initialize TodoManager with a JSON file for persistence.
        
        Args:
            file_path (str): Path to the JSON file for storing tasks
        """
        self.file_path = Path(file_path)
        self.tasks: Dict[str, Task] = {}
        self.load_tasks()
    
    def load_tasks(self) -> None:
        """Load tasks from the JSON file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for task_id, task_data in data.items():
                        self.tasks[task_id] = Task.from_dict(task_data)
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not load tasks from {self.file_path}: {e}")
                self.tasks = {}
    
    def save_tasks(self) -> None:
        """Save all tasks to the JSON file."""
        try:
            # Create parent directories if they don't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {task_id: task.to_dict() for task_id, task in self.tasks.items()}
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except (OSError, IOError) as e:
            print(f"Error: Could not save tasks to {self.file_path}: {e}")
            raise
    
    def add_task(self, 
                 title: str,
                 description: Optional[str] = None,
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new task to the manager.
        
        Args:
            title (str): Title of the task
            description (Optional[str]): Detailed description
            priority (TaskPriority): Priority level of the task
            tags (Optional[List[str]]): List of tags
            metadata (Optional[Dict[str, Any]]): Additional metadata
            
        Returns:
            str: The unique ID of the created task
        """
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        self.save_tasks()
        return task_id
    
    def list_tasks(self, 
                   status_filter: Optional[TaskStatus] = None,
                   priority_filter: Optional[TaskPriority] = None,
                   tag_filter: Optional[str] = None) -> List[Task]:
        """
        List tasks with optional filters.
        
        Args:
            status_filter (Optional[TaskStatus]): Filter by task status
            priority_filter (Optional[TaskPriority]): Filter by task priority
            tag_filter (Optional[str]): Filter by tag (exact match)
            
        Returns:
            List[Task]: Filtered list of tasks
        """
        tasks = list(self.tasks.values())
        
        if status_filter is not None:
            tasks = [task for task in tasks if task.status == status_filter]
        
        if priority_filter is not None:
            tasks = [task for task in tasks if task.priority == priority_filter]
        
        if tag_filter is not None:
            tasks = [task for task in tasks if tag_filter in task.tags]
        
        # Sort by priority (CRITICAL -> HIGH -> MEDIUM -> LOW) then by created_at
        priority_order = {TaskPriority.CRITICAL: 0, TaskPriority.HIGH: 1, 
                         TaskPriority.MEDIUM: 2, TaskPriority.LOW: 3}
        
        return sorted(tasks, key=lambda t: (priority_order[t.priority], t.created_at))
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a specific task by ID.
        
        Args:
            task_id (str): The unique ID of the task
            
        Returns:
            Optional[Task]: The task if found, None otherwise
        """
        return self.tasks.get(task_id)
    
    def complete_task(self, task_id: str) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id (str): The unique ID of the task to complete
            
        Returns:
            bool: True if task was found and completed, False otherwise
        """
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].updated_at = datetime.now()
            self.save_tasks()
            return True
        return False
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from the manager.
        
        Args:
            task_id (str): The unique ID of the task to delete
            
        Returns:
            bool: True if task was found and deleted, False otherwise
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.save_tasks()
            return True
        return False
    
    def update_task(self,
                    task_id: str,
                    title: Optional[str] = None,
                    description: Optional[str] = None,
                    priority: Optional[TaskPriority] = None,
                    status: Optional[TaskStatus] = None,
                    tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing task.
        
        Args:
            task_id (str): The unique ID of the task to update
            title (Optional[str]): New title
            description (Optional[str]): New description
            priority (Optional[TaskPriority]): New priority
            status (Optional[TaskStatus]): New status
            tags (Optional[List[str]]): New tags list
            metadata (Optional[Dict[str, Any]]): New metadata
            
        Returns:
            bool: True if task was found and updated, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if title is not None:
            task.title = title
        if description is not None:
            task.description = description
        if priority is not None:
            task.priority = priority
        if status is not None:
            task.status = status
        if tags is not None:
            task.tags = tags
        if metadata is not None:
            task.metadata = metadata
        
        task.updated_at = datetime.now()
        self.save_tasks()
        return True
    
    def search_tasks(self, query: str) -> List[Task]:
        """
        Search tasks by title or description.
        
        Args:
            query (str): Search query string
            
        Returns:
            List[Task]: List of tasks matching the search query
        """
        query_lower = query.lower()
        matching_tasks = []
        
        for task in self.tasks.values():
            if (query_lower in task.title.lower() or 
                (task.description and query_lower in task.description.lower())):
                matching_tasks.append(task)
        
        # Sort by relevance (title match first, then description match)
        title_matches = [t for t in matching_tasks if query_lower in t.title.lower()]
        desc_matches = [t for t in matching_tasks if t not in title_matches]
        
        return title_matches + desc_matches
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tasks.
        
        Returns:
            Dict[str, Any]: Dictionary containing task statistics
        """
        if not self.tasks:
            return {
                'total': 0,
                'by_status': {},
                'by_priority': {},
                'completion_rate': 0.0
            }
        
        status_counts = {}
        priority_counts = {}
        
        for task in self.tasks.values():
            status_counts[task.status.name] = status_counts.get(task.status.name, 0) + 1
            priority_counts[task.priority.name] = priority_counts.get(task.priority.name, 0) + 1
        
        completed_count = status_counts.get('COMPLETED', 0)
        total_count = len(self.tasks)
        completion_rate = (completed_count / total_count) * 100 if total_count > 0 else 0.0
        
        return {
            'total': total_count,
            'by_status': status_counts,
            'by_priority': priority_counts,
            'completion_rate': round(completion_rate, 2)
        }
    
    def clear_completed_tasks(self) -> int:
        """
        Remove all completed tasks.
        
        Returns:
            int: Number of tasks removed
        """
        completed_ids = [task_id for task_id, task in self.tasks.items() 
                        if task.status == TaskStatus.COMPLETED]
        
        for task_id in completed_ids:
            del self.tasks[task_id]
        
        if completed_ids:
            self.save_tasks()
        
        return len(completed_ids)
    
    def export_to_json(self, export_path: str) -> None:
        """
        Export all tasks to a different JSON file.
        
        Args:
            export_path (str): Path for the export file
        """
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def import_from_json(self, import_path: str, merge: bool = False) -> int:
        """
        Import tasks from a JSON file.
        
        Args:
            import_path (str): Path to the import file
            merge (bool): If True, merge with existing tasks. If False, replace all tasks.
            
        Returns:
            int: Number of tasks imported
        """
        import_file = Path(import_path)
        if not import_file.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")
        
        with open(import_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not merge:
            self.tasks = {}
        
        imported_count = 0
        for task_id, task_data in data.items():
            try:
                task = Task.from_dict(task_data)
                # Generate new ID if merging and ID already exists
                if merge and task_id in self.tasks:
                    task_id = str(uuid.uuid4())
                    task.id = task_id
                
                self.tasks[task_id] = task
                imported_count += 1
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not import task {task_id}: {e}")
        
        self.save_tasks()
        return imported_count