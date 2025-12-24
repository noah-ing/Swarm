from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from typing import Optional, Any
import json
from datetime import datetime

class TaskPriority(Enum):
    """Enumeration of task priority levels"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class TaskStatus(Enum):
    """Enumeration of task completion statuses"""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    BLOCKED = auto()
    CANCELLED = auto()

@dataclass
class Task:
    """
    Core Task data model representing a task with priority, status, and metadata
    
    Attributes:
        id (Optional[str]): Unique identifier for the task
        title (str): Brief description of the task
        description (Optional[str]): Detailed explanation of the task
        priority (TaskPriority): Priority level of the task
        status (TaskStatus): Current status of the task
        created_at (datetime): Timestamp of task creation
        updated_at (Optional[datetime]): Timestamp of last task update
        tags (list[str]): List of tags associated with the task
        metadata (dict[str, Any]): Additional metadata for the task
    """
    id: Optional[str] = None
    title: str = "Untitled Task"
    description: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert task to a dictionary representation
        
        Returns:
            dict: Serializable dictionary of the task
        """
        task_dict = asdict(self)
        task_dict['priority'] = self.priority.name
        task_dict['status'] = self.status.name
        task_dict['created_at'] = self.created_at.isoformat()
        task_dict['updated_at'] = self.updated_at.isoformat() if self.updated_at else None
        return task_dict

    def to_json(self) -> str:
        """
        Serialize task to JSON string
        
        Returns:
            str: JSON representation of the task
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        """
        Create a Task instance from a dictionary
        
        Args:
            data (dict): Dictionary containing task data
        
        Returns:
            Task: Reconstructed Task instance
        """
        # Convert priority and status back to Enum
        if 'priority' in data:
            data['priority'] = TaskPriority[data['priority']]
        if 'status' in data:
            data['status'] = TaskStatus[data['status']]
        
        # Convert datetime strings back to datetime objects
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Task':
        """
        Create a Task instance from a JSON string
        
        Args:
            json_str (str): JSON string representing a task
        
        Returns:
            Task: Reconstructed Task instance
        """
        return cls.from_dict(json.loads(json_str))