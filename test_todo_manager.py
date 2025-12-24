import json
import os
import tempfile
from pathlib import Path
import pytest
from datetime import datetime

from todo_manager import TodoManager
from task import Task, TaskPriority, TaskStatus


class TestTodoManager:
    """Test suite for TodoManager class"""
    
    def setup_method(self):
        """Set up test with temporary file"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.todo_manager = TodoManager(self.temp_file.name)
    
    def teardown_method(self):
        """Clean up temporary file"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """Test TodoManager initialization"""
        assert self.todo_manager.file_path.name.endswith('.json')
        assert isinstance(self.todo_manager.tasks, dict)
        assert len(self.todo_manager.tasks) == 0
    
    def test_add_task_basic(self):
        """Test adding a basic task"""
        task_id = self.todo_manager.add_task("Test Task")
        
        assert task_id in self.todo_manager.tasks
        task = self.todo_manager.tasks[task_id]
        assert task.title == "Test Task"
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.id == task_id
    
    def test_add_task_with_details(self):
        """Test adding a task with all details"""
        task_id = self.todo_manager.add_task(
            title="Detailed Task",
            description="A task with full details",
            priority=TaskPriority.HIGH,
            tags=["work", "urgent"],
            metadata={"project": "test", "estimate": 2}
        )
        
        task = self.todo_manager.tasks[task_id]
        assert task.title == "Detailed Task"
        assert task.description == "A task with full details"
        assert task.priority == TaskPriority.HIGH
        assert task.tags == ["work", "urgent"]
        assert task.metadata == {"project": "test", "estimate": 2}
    
    def test_list_tasks_empty(self):
        """Test listing tasks when no tasks exist"""
        tasks = self.todo_manager.list_tasks()
        assert tasks == []
    
    def test_list_tasks_with_tasks(self):
        """Test listing tasks"""
        # Add multiple tasks
        id1 = self.todo_manager.add_task("Task 1", priority=TaskPriority.HIGH)
        id2 = self.todo_manager.add_task("Task 2", priority=TaskPriority.LOW)
        id3 = self.todo_manager.add_task("Task 3", priority=TaskPriority.CRITICAL)
        
        tasks = self.todo_manager.list_tasks()
        assert len(tasks) == 3
        
        # Should be sorted by priority (CRITICAL first)
        assert tasks[0].priority == TaskPriority.CRITICAL
        assert tasks[1].priority == TaskPriority.HIGH
        assert tasks[2].priority == TaskPriority.LOW
    
    def test_list_tasks_with_status_filter(self):
        """Test listing tasks with status filter"""
        id1 = self.todo_manager.add_task("Pending Task")
        id2 = self.todo_manager.add_task("Another Task")
        self.todo_manager.complete_task(id2)
        
        pending_tasks = self.todo_manager.list_tasks(status_filter=TaskStatus.PENDING)
        completed_tasks = self.todo_manager.list_tasks(status_filter=TaskStatus.COMPLETED)
        
        assert len(pending_tasks) == 1
        assert len(completed_tasks) == 1
        assert pending_tasks[0].id == id1
        assert completed_tasks[0].id == id2
    
    def test_list_tasks_with_priority_filter(self):
        """Test listing tasks with priority filter"""
        id1 = self.todo_manager.add_task("High Task", priority=TaskPriority.HIGH)
        id2 = self.todo_manager.add_task("Low Task", priority=TaskPriority.LOW)
        
        high_tasks = self.todo_manager.list_tasks(priority_filter=TaskPriority.HIGH)
        low_tasks = self.todo_manager.list_tasks(priority_filter=TaskPriority.LOW)
        
        assert len(high_tasks) == 1
        assert len(low_tasks) == 1
        assert high_tasks[0].id == id1
        assert low_tasks[0].id == id2
    
    def test_list_tasks_with_tag_filter(self):
        """Test listing tasks with tag filter"""
        id1 = self.todo_manager.add_task("Work Task", tags=["work", "urgent"])
        id2 = self.todo_manager.add_task("Personal Task", tags=["personal"])
        
        work_tasks = self.todo_manager.list_tasks(tag_filter="work")
        personal_tasks = self.todo_manager.list_tasks(tag_filter="personal")
        urgent_tasks = self.todo_manager.list_tasks(tag_filter="urgent")
        
        assert len(work_tasks) == 1
        assert len(personal_tasks) == 1
        assert len(urgent_tasks) == 1
        assert work_tasks[0].id == id1
        assert personal_tasks[0].id == id2
    
    def test_get_task(self):
        """Test getting a specific task"""
        task_id = self.todo_manager.add_task("Test Task")
        
        task = self.todo_manager.get_task(task_id)
        assert task is not None
        assert task.title == "Test Task"
        
        # Test non-existent task
        non_existent = self.todo_manager.get_task("fake-id")
        assert non_existent is None
    
    def test_complete_task(self):
        """Test completing a task"""
        task_id = self.todo_manager.add_task("Task to Complete")
        
        result = self.todo_manager.complete_task(task_id)
        assert result is True
        
        task = self.todo_manager.tasks[task_id]
        assert task.status == TaskStatus.COMPLETED
        assert task.updated_at is not None
        
        # Test completing non-existent task
        result = self.todo_manager.complete_task("fake-id")
        assert result is False
    
    def test_delete_task(self):
        """Test deleting a task"""
        task_id = self.todo_manager.add_task("Task to Delete")
        
        result = self.todo_manager.delete_task(task_id)
        assert result is True
        assert task_id not in self.todo_manager.tasks
        
        # Test deleting non-existent task
        result = self.todo_manager.delete_task("fake-id")
        assert result is False
    
    def test_update_task(self):
        """Test updating a task"""
        task_id = self.todo_manager.add_task(
            "Original Title", 
            description="Original description",
            priority=TaskPriority.LOW
        )
        
        result = self.todo_manager.update_task(
            task_id,
            title="Updated Title",
            description="Updated description",
            priority=TaskPriority.HIGH,
            status=TaskStatus.IN_PROGRESS,
            tags=["updated"],
            metadata={"updated": True}
        )
        
        assert result is True
        
        task = self.todo_manager.tasks[task_id]
        assert task.title == "Updated Title"
        assert task.description == "Updated description"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.tags == ["updated"]
        assert task.metadata == {"updated": True}
        assert task.updated_at is not None
        
        # Test updating non-existent task
        result = self.todo_manager.update_task("fake-id", title="New Title")
        assert result is False
    
    def test_search_tasks(self):
        """Test searching tasks"""
        id1 = self.todo_manager.add_task("Buy groceries", description="Milk, eggs, bread")
        id2 = self.todo_manager.add_task("Write report", description="Annual financial report")
        id3 = self.todo_manager.add_task("Call dentist", description="Schedule appointment")
        
        # Search by title
        results = self.todo_manager.search_tasks("buy")
        assert len(results) == 1
        assert results[0].id == id1
        
        # Search by description
        results = self.todo_manager.search_tasks("report")
        assert len(results) == 1
        assert results[0].id == id2
        
        # Search with multiple matches
        results = self.todo_manager.search_tasks("e")  # Should match multiple tasks
        assert len(results) >= 2
        
        # Case insensitive search
        results = self.todo_manager.search_tasks("BUY")
        assert len(results) == 1
        assert results[0].id == id1
    
    def test_get_stats_empty(self):
        """Test getting stats with no tasks"""
        stats = self.todo_manager.get_stats()
        
        assert stats['total'] == 0
        assert stats['by_status'] == {}
        assert stats['by_priority'] == {}
        assert stats['completion_rate'] == 0.0
    
    def test_get_stats_with_tasks(self):
        """Test getting stats with tasks"""
        id1 = self.todo_manager.add_task("Task 1", priority=TaskPriority.HIGH)
        id2 = self.todo_manager.add_task("Task 2", priority=TaskPriority.HIGH)
        id3 = self.todo_manager.add_task("Task 3", priority=TaskPriority.LOW)
        
        self.todo_manager.complete_task(id1)
        
        stats = self.todo_manager.get_stats()
        
        assert stats['total'] == 3
        assert stats['by_status']['PENDING'] == 2
        assert stats['by_status']['COMPLETED'] == 1
        assert stats['by_priority']['HIGH'] == 2
        assert stats['by_priority']['LOW'] == 1
        assert stats['completion_rate'] == 33.33
    
    def test_clear_completed_tasks(self):
        """Test clearing completed tasks"""
        id1 = self.todo_manager.add_task("Task 1")
        id2 = self.todo_manager.add_task("Task 2")
        id3 = self.todo_manager.add_task("Task 3")
        
        self.todo_manager.complete_task(id1)
        self.todo_manager.complete_task(id2)
        
        cleared_count = self.todo_manager.clear_completed_tasks()
        
        assert cleared_count == 2
        assert len(self.todo_manager.tasks) == 1
        assert id3 in self.todo_manager.tasks
        assert id1 not in self.todo_manager.tasks
        assert id2 not in self.todo_manager.tasks
    
    def test_persistence_save_and_load(self):
        """Test saving and loading tasks from file"""
        # Add some tasks
        id1 = self.todo_manager.add_task("Persistent Task 1", priority=TaskPriority.HIGH)
        id2 = self.todo_manager.add_task("Persistent Task 2", tags=["test"])
        
        # Create a new manager with the same file
        new_manager = TodoManager(self.temp_file.name)
        
        # Verify tasks were loaded
        assert len(new_manager.tasks) == 2
        assert id1 in new_manager.tasks
        assert id2 in new_manager.tasks
        assert new_manager.tasks[id1].title == "Persistent Task 1"
        assert new_manager.tasks[id1].priority == TaskPriority.HIGH
        assert new_manager.tasks[id2].tags == ["test"]
    
    def test_export_import_json(self):
        """Test exporting and importing tasks"""
        # Add some tasks
        id1 = self.todo_manager.add_task("Export Task 1")
        id2 = self.todo_manager.add_task("Export Task 2", priority=TaskPriority.CRITICAL)
        
        # Export to temporary file
        export_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_export.json')
        export_file.close()
        
        try:
            self.todo_manager.export_to_json(export_file.name)
            
            # Create new manager and import
            new_manager = TodoManager()
            imported_count = new_manager.import_from_json(export_file.name, merge=False)
            
            assert imported_count == 2
            assert len(new_manager.tasks) == 2
            
            # Verify tasks were imported correctly
            imported_tasks = list(new_manager.tasks.values())
            titles = [task.title for task in imported_tasks]
            assert "Export Task 1" in titles
            assert "Export Task 2" in titles
            
        finally:
            os.unlink(export_file.name)
    
    def test_import_merge(self):
        """Test importing with merge option"""
        # Add initial task
        self.todo_manager.add_task("Original Task")
        
        # Create export data
        export_data = {
            "task_1": {
                "id": "task_1",
                "title": "Imported Task",
                "description": None,
                "priority": "MEDIUM",
                "status": "PENDING",
                "created_at": datetime.now().isoformat(),
                "updated_at": None,
                "tags": [],
                "metadata": {}
            }
        }
        
        # Write export data to file
        import_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_import.json')
        json.dump(export_data, import_file)
        import_file.close()
        
        try:
            # Import with merge
            imported_count = self.todo_manager.import_from_json(import_file.name, merge=True)
            
            assert imported_count == 1
            assert len(self.todo_manager.tasks) == 2  # Original + imported
            
            # Verify both tasks exist
            titles = [task.title for task in self.todo_manager.tasks.values()]
            assert "Original Task" in titles
            assert "Imported Task" in titles
            
        finally:
            os.unlink(import_file.name)


def test_todo_manager_integration():
    """Integration test demonstrating full TodoManager workflow"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        temp_file.close()
        
        try:
            manager = TodoManager(temp_file.name)
            
            # Add various tasks
            task1_id = manager.add_task(
                "Complete project proposal",
                description="Write and review the Q4 project proposal",
                priority=TaskPriority.HIGH,
                tags=["work", "deadline"]
            )
            
            task2_id = manager.add_task(
                "Buy birthday gift",
                description="Find a gift for mom's birthday",
                priority=TaskPriority.MEDIUM,
                tags=["personal", "family"]
            )
            
            task3_id = manager.add_task(
                "Fix critical bug",
                priority=TaskPriority.CRITICAL,
                tags=["work", "urgent"]
            )
            
            # Test filtering and searching
            high_priority_tasks = manager.list_tasks(priority_filter=TaskPriority.HIGH)
            assert len(high_priority_tasks) == 1
            
            work_tasks = manager.list_tasks(tag_filter="work")
            assert len(work_tasks) == 2
            
            search_results = manager.search_tasks("project")
            assert len(search_results) == 1
            
            # Complete some tasks
            manager.complete_task(task1_id)
            manager.update_task(task2_id, status=TaskStatus.IN_PROGRESS)
            
            # Check stats
            stats = manager.get_stats()
            assert stats['total'] == 3
            assert stats['by_status']['COMPLETED'] == 1
            assert stats['by_status']['IN_PROGRESS'] == 1
            assert stats['by_status']['PENDING'] == 1
            
            # Verify persistence by creating new manager
            new_manager = TodoManager(temp_file.name)
            assert len(new_manager.tasks) == 3
            assert new_manager.get_task(task1_id).status == TaskStatus.COMPLETED
            
        finally:
            os.unlink(temp_file.name)


if __name__ == "__main__":
    # Run basic tests
    test_todo_manager_integration()
    print("Integration test passed!")