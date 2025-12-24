import json
import pytest
from datetime import datetime
from task import Task, TaskPriority, TaskStatus

def test_task_default_values():
    """Test task creation with default values"""
    task = Task()
    
    assert task.id is None
    assert task.title == "Untitled Task"
    assert task.description is None
    assert task.priority == TaskPriority.MEDIUM
    assert task.status == TaskStatus.PENDING
    assert isinstance(task.created_at, datetime)
    assert task.updated_at is None
    assert task.tags == []
    assert task.metadata == {}

def test_task_all_priority_levels():
    """Test all priority levels"""
    priorities = [TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH, TaskPriority.CRITICAL]
    
    for priority in priorities:
        task = Task(title=f"Task with {priority.name} priority", priority=priority)
        assert task.priority == priority
        
        # Test serialization preserves priority
        json_str = task.to_json()
        restored_task = Task.from_json(json_str)
        assert restored_task.priority == priority

def test_task_all_status_levels():
    """Test all status levels"""
    statuses = [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, 
                TaskStatus.BLOCKED, TaskStatus.CANCELLED]
    
    for status in statuses:
        task = Task(title=f"Task with {status.name} status", status=status)
        assert task.status == status
        
        # Test serialization preserves status
        json_str = task.to_json()
        restored_task = Task.from_json(json_str)
        assert restored_task.status == status

def test_task_empty_json_serialization():
    """Test JSON serialization with minimal data"""
    task = Task()
    
    json_str = task.to_json()
    restored_task = Task.from_json(json_str)
    
    assert restored_task.title == task.title
    assert restored_task.priority == task.priority
    assert restored_task.status == task.status
    assert restored_task.created_at == task.created_at

def test_task_complex_metadata():
    """Test task with complex metadata structures"""
    complex_metadata = {
        "nested": {
            "level1": {
                "level2": "deep value"
            }
        },
        "list": [1, 2, 3, "mixed", {"type": "object"}],
        "boolean": True,
        "null_value": None,
        "number": 42.5
    }
    
    task = Task(
        title="Complex Metadata Task",
        metadata=complex_metadata
    )
    
    # Test serialization preserves complex metadata
    json_str = task.to_json()
    restored_task = Task.from_json(json_str)
    
    assert restored_task.metadata == complex_metadata

def test_task_datetime_handling():
    """Test datetime serialization and deserialization"""
    now = datetime.now()
    task = Task(title="DateTime Test")
    task.updated_at = now
    
    json_str = task.to_json()
    restored_task = Task.from_json(json_str)
    
    # Datetime objects should be equal (within microsecond precision)
    assert restored_task.created_at == task.created_at
    assert restored_task.updated_at == task.updated_at

def test_task_unicode_handling():
    """Test task with unicode characters"""
    task = Task(
        title="Unicode Task 🚀",
        description="Testing with émojis and spéciál characters: 你好",
        tags=["unicode", "test", "🏷️"],
        metadata={"emoji": "🎯", "chinese": "测试"}
    )
    
    json_str = task.to_json()
    restored_task = Task.from_json(json_str)
    
    assert restored_task.title == task.title
    assert restored_task.description == task.description
    assert restored_task.tags == task.tags
    assert restored_task.metadata == task.metadata

def test_task_large_data():
    """Test task with large amounts of data"""
    large_list = [f"item_{i}" for i in range(1000)]
    large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
    
    task = Task(
        title="Large Data Task",
        tags=large_list[:10],  # Reasonable tag count
        metadata=large_metadata
    )
    
    json_str = task.to_json()
    restored_task = Task.from_json(json_str)
    
    assert restored_task.tags == task.tags
    assert restored_task.metadata == task.metadata

def test_task_to_dict_format():
    """Test the exact format of to_dict method"""
    task = Task(
        id="test_id",
        title="Dict Format Test",
        priority=TaskPriority.HIGH,
        status=TaskStatus.COMPLETED
    )
    
    task_dict = task.to_dict()
    
    # Verify enum values are converted to strings
    assert task_dict['priority'] == "HIGH"
    assert task_dict['status'] == "COMPLETED"
    
    # Verify datetime is converted to ISO format string
    assert isinstance(task_dict['created_at'], str)
    assert 'T' in task_dict['created_at']  # ISO format contains 'T'

def test_json_roundtrip_fidelity():
    """Test complete fidelity of JSON serialization/deserialization"""
    original_task = Task(
        id="roundtrip_test",
        title="JSON Roundtrip Test",
        description="Testing complete data preservation",
        priority=TaskPriority.CRITICAL,
        status=TaskStatus.BLOCKED,
        tags=["test", "json", "roundtrip"],
        metadata={"version": 2.1, "validated": True, "components": ["a", "b", "c"]}
    )
    
    # Multiple roundtrips should preserve data
    for i in range(3):
        json_str = original_task.to_json()
        original_task = Task.from_json(json_str)
    
    # Final verification
    assert original_task.id == "roundtrip_test"
    assert original_task.title == "JSON Roundtrip Test"
    assert original_task.priority == TaskPriority.CRITICAL
    assert original_task.status == TaskStatus.BLOCKED
    assert original_task.tags == ["test", "json", "roundtrip"]
    assert original_task.metadata["version"] == 2.1
    assert original_task.metadata["validated"] is True
    assert original_task.metadata["components"] == ["a", "b", "c"]