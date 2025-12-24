import json
from datetime import datetime
from task import Task, TaskPriority, TaskStatus

def test_task_creation():
    """Test basic task creation"""
    task = Task(
        title="Test Task",
        description="A task for testing",
        priority=TaskPriority.HIGH,
        status=TaskStatus.IN_PROGRESS,
        tags=["test", "development"]
    )
    
    assert task.title == "Test Task"
    assert task.priority == TaskPriority.HIGH
    assert task.status == TaskStatus.IN_PROGRESS
    assert task.tags == ["test", "development"]
    assert isinstance(task.created_at, datetime)

def test_task_json_serialization():
    """Test JSON serialization and deserialization"""
    original_task = Task(
        id="task_123",
        title="Serialize Me",
        priority=TaskPriority.CRITICAL,
        status=TaskStatus.BLOCKED,
        tags=["serialization", "testing"]
    )
    
    # Convert to JSON and back
    json_str = original_task.to_json()
    reconstructed_task = Task.from_json(json_str)
    
    # Verify all attributes match
    assert reconstructed_task.id == original_task.id
    assert reconstructed_task.title == original_task.title
    assert reconstructed_task.priority == original_task.priority
    assert reconstructed_task.status == original_task.status
    assert reconstructed_task.tags == original_task.tags

def test_task_dict_conversion():
    """Test dictionary conversion methods"""
    task = Task(
        title="Dict Test",
        priority=TaskPriority.LOW,
        status=TaskStatus.COMPLETED
    )
    
    # Convert to dict
    task_dict = task.to_dict()
    
    # Verify dictionary contents
    assert task_dict['title'] == "Dict Test"
    assert task_dict['priority'] == "LOW"
    assert task_dict['status'] == "COMPLETED"

def test_task_metadata():
    """Test adding metadata to a task"""
    task = Task(
        title="Metadata Task",
        metadata={
            "project": "SwarmAI",
            "complexity": 5
        }
    )
    
    assert task.metadata['project'] == "SwarmAI"
    assert task.metadata['complexity'] == 5

def test_task_update():
    """Test task update functionality"""
    task = Task(title="Initial Task")
    original_created_at = task.created_at
    
    task.status = TaskStatus.IN_PROGRESS
    task.updated_at = datetime.now()
    
    assert task.status == TaskStatus.IN_PROGRESS
    assert task.updated_at is not None
    assert task.created_at == original_created_at