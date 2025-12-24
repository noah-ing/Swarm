#!/usr/bin/env python3
"""
Test suite for TodoCLI - comprehensive tests for the command-line interface.

Tests all CLI commands, argument parsing, error handling, and output formatting.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

from todo_cli import TodoCLI
from todo_manager import TodoManager
from task import Task, TaskPriority, TaskStatus


class TestTodoCLI:
    """Test suite for TodoCLI class."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'test_todos.json')
        self.cli = TodoCLI()
        
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_setup_parser(self):
        """Test argument parser setup."""
        parser = self.cli.setup_parser()
        
        # Test basic parser structure
        assert parser.prog == 'todo'
        assert 'todo task management' in parser.description
        
        # Test that all commands are available
        help_text = parser.format_help()
        commands = ['add', 'list', 'complete', 'delete', 'update', 'search', 'show', 'stats', 'clear', 'export', 'import']
        for command in commands:
            assert command in help_text
    
    def test_init_manager(self):
        """Test TodoManager initialization."""
        self.cli.init_manager(self.temp_file)
        assert isinstance(self.cli.manager, TodoManager)
        assert self.cli.manager.file_path == Path(self.temp_file)
    
    def test_parse_priority(self):
        """Test priority parsing."""
        assert self.cli.parse_priority('low') == TaskPriority.LOW
        assert self.cli.parse_priority('medium') == TaskPriority.MEDIUM
        assert self.cli.parse_priority('high') == TaskPriority.HIGH
        assert self.cli.parse_priority('critical') == TaskPriority.CRITICAL
        assert self.cli.parse_priority('HIGH') == TaskPriority.HIGH  # Case insensitive
    
    def test_parse_status(self):
        """Test status parsing."""
        assert self.cli.parse_status('pending') == TaskStatus.PENDING
        assert self.cli.parse_status('in_progress') == TaskStatus.IN_PROGRESS
        assert self.cli.parse_status('completed') == TaskStatus.COMPLETED
        assert self.cli.parse_status('blocked') == TaskStatus.BLOCKED
        assert self.cli.parse_status('cancelled') == TaskStatus.CANCELLED
    
    def test_parse_tags(self):
        """Test tags parsing."""
        assert self.cli.parse_tags('tag1,tag2,tag3') == ['tag1', 'tag2', 'tag3']
        assert self.cli.parse_tags('  tag1  ,  tag2  ') == ['tag1', 'tag2']
        assert self.cli.parse_tags('single') == ['single']
        assert self.cli.parse_tags('') == []
    
    def test_parse_metadata(self):
        """Test metadata parsing."""
        metadata = self.cli.parse_metadata('{"key": "value", "num": 42}')
        assert metadata == {"key": "value", "num": 42}
        
        # Test invalid JSON
        with patch('sys.stderr'):
            with pytest.raises(SystemExit):
                self.cli.parse_metadata('invalid json')
    
    def test_format_task_table(self):
        """Test table formatting."""
        # Empty list
        output = self.cli.format_task_table([])
        assert "No tasks found" in output
        
        # With tasks
        task = Task(
            id='test-id',
            title='Test Task',
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH
        )
        output = self.cli.format_task_table([task])
        assert 'Test Task' in output
        assert 'PENDING' in output
        assert 'HIGH' in output
        assert 'test-id'[:8] in output
    
    def test_format_task_brief(self):
        """Test brief formatting."""
        task1 = Task(
            id='test-id-1',
            title='Pending Task',
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH
        )
        task2 = Task(
            id='test-id-2',
            title='Completed Task',
            status=TaskStatus.COMPLETED,
            priority=TaskPriority.LOW
        )
        
        output = self.cli.format_task_brief([task1, task2])
        assert '○ test-id' in output  # Pending task
        assert '✓ test-id' in output  # Completed task
        assert '!' in output  # High priority indicator
    
    def test_format_task_json(self):
        """Test JSON formatting."""
        task = Task(
            id='test-id',
            title='Test Task',
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH
        )
        
        output = self.cli.format_task_json([task])
        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]['title'] == 'Test Task'
        assert parsed[0]['status'] == 'PENDING'
        assert parsed[0]['priority'] == 'HIGH'
    
    def test_format_task_details(self):
        """Test detailed task formatting."""
        task = Task(
            id='test-id',
            title='Test Task',
            description='Test description',
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            tags=['tag1', 'tag2'],
            metadata={'key': 'value'}
        )
        
        output = self.cli.format_task_details(task)
        assert 'Test Task' in output
        assert 'Test description' in output
        assert 'PENDING' in output
        assert 'HIGH' in output
        assert 'tag1, tag2' in output
        assert "{'key': 'value'}" in output
    
    @patch('builtins.input', return_value='y')
    def test_confirm_action_yes(self, mock_input):
        """Test confirmation with yes response."""
        assert self.cli.confirm_action("Test message?") is True
        mock_input.assert_called_once_with("Test message? (y/N): ")
    
    @patch('builtins.input', return_value='n')
    def test_confirm_action_no(self, mock_input):
        """Test confirmation with no response."""
        assert self.cli.confirm_action("Test message?") is False
    
    @patch('builtins.input', side_effect=KeyboardInterrupt)
    def test_confirm_action_keyboard_interrupt(self, mock_input):
        """Test confirmation with keyboard interrupt."""
        assert self.cli.confirm_action("Test message?") is False
    
    def test_cmd_add(self):
        """Test add command."""
        self.cli.init_manager(self.temp_file)
        
        # Mock arguments
        args = MagicMock()
        args.title = 'Test Task'
        args.description = 'Test description'
        args.priority = 'high'
        args.tags = 'tag1,tag2'
        args.metadata = '{"key": "value"}'
        
        # Capture output
        with patch('builtins.print') as mock_print:
            self.cli.cmd_add(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert '✓' in output
            assert 'Task added' in output
        
        # Verify task was added
        tasks = self.cli.manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == 'Test Task'
        assert tasks[0].priority == TaskPriority.HIGH
        assert 'tag1' in tasks[0].tags
    
    def test_cmd_list(self):
        """Test list command."""
        self.cli.init_manager(self.temp_file)
        
        # Add test tasks
        self.cli.manager.add_task('Task 1', priority=TaskPriority.HIGH)
        self.cli.manager.add_task('Task 2', priority=TaskPriority.LOW)
        
        # Test list all
        args = MagicMock()
        args.status = None
        args.priority = None
        args.tag = None
        args.format = 'table'
        args.limit = None
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_list(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert 'Task 1' in output
            assert 'Task 2' in output
    
    def test_cmd_complete(self):
        """Test complete command."""
        self.cli.init_manager(self.temp_file)
        
        # Add test task
        task_id = self.cli.manager.add_task('Test Task')
        
        # Complete task
        args = MagicMock()
        args.task_ids = [task_id]
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_complete(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert '✓' in output
            assert 'Completed 1 task' in output
        
        # Verify task was completed
        task = self.cli.manager.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
    
    def test_cmd_delete(self):
        """Test delete command."""
        self.cli.init_manager(self.temp_file)
        
        # Add test task
        task_id = self.cli.manager.add_task('Test Task')
        
        # Delete task with force
        args = MagicMock()
        args.task_ids = [task_id]
        args.force = True
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_delete(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert '✓' in output
            assert 'Deleted 1 task' in output
        
        # Verify task was deleted
        assert self.cli.manager.get_task(task_id) is None
    
    @patch('builtins.input', return_value='n')
    def test_cmd_delete_cancelled(self, mock_input):
        """Test delete command cancelled by user."""
        self.cli.init_manager(self.temp_file)
        
        # Add test task
        task_id = self.cli.manager.add_task('Test Task')
        
        # Try to delete without force
        args = MagicMock()
        args.task_ids = [task_id]
        args.force = False
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_delete(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert 'cancelled' in output.lower()
        
        # Verify task was not deleted
        assert self.cli.manager.get_task(task_id) is not None
    
    def test_cmd_update(self):
        """Test update command."""
        self.cli.init_manager(self.temp_file)
        
        # Add test task
        task_id = self.cli.manager.add_task('Original Title')
        
        # Update task
        args = MagicMock()
        args.task_id = task_id
        args.title = 'Updated Title'
        args.description = 'Updated description'
        args.priority = 'critical'
        args.status = 'in_progress'
        args.tags = 'new_tag'
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_update(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert '✓' in output
            assert 'updated' in output
        
        # Verify task was updated
        task = self.cli.manager.get_task(task_id)
        assert task.title == 'Updated Title'
        assert task.description == 'Updated description'
        assert task.priority == TaskPriority.CRITICAL
        assert task.status == TaskStatus.IN_PROGRESS
        assert 'new_tag' in task.tags
    
    def test_cmd_search(self):
        """Test search command."""
        self.cli.init_manager(self.temp_file)
        
        # Add test tasks
        self.cli.manager.add_task('Buy groceries')
        self.cli.manager.add_task('Walk the dog')
        
        # Search for tasks
        args = MagicMock()
        args.query = 'groceries'
        args.format = 'table'
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_search(args)
            # Should be called twice: once for results, once for info
            assert mock_print.call_count >= 1
            
            # Check if groceries task was found in output
            calls = mock_print.call_args_list
            output = ' '.join(str(call[0][0]) for call in calls)
            assert 'groceries' in output.lower()
    
    def test_cmd_show(self):
        """Test show command."""
        self.cli.init_manager(self.temp_file)
        
        # Add test task
        task_id = self.cli.manager.add_task('Test Task', description='Test description')
        
        # Show task details
        args = MagicMock()
        args.task_id = task_id
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_show(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert 'Test Task' in output
            assert 'Test description' in output
            assert 'Task Details' in output
    
    def test_cmd_show_not_found(self):
        """Test show command with non-existent task."""
        self.cli.init_manager(self.temp_file)
        
        args = MagicMock()
        args.task_id = 'non-existent'
        
        with patch('sys.stderr'), pytest.raises(SystemExit):
            self.cli.cmd_show(args)
    
    def test_cmd_stats(self):
        """Test stats command."""
        self.cli.init_manager(self.temp_file)
        
        # Add test tasks
        task_id1 = self.cli.manager.add_task('Task 1', priority=TaskPriority.HIGH)
        self.cli.manager.add_task('Task 2', priority=TaskPriority.LOW)
        self.cli.manager.complete_task(task_id1)
        
        # Get stats
        args = MagicMock()
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_stats(args)
            mock_print.assert_called()
            
            # Combine all print calls to check output
            calls = mock_print.call_args_list
            output = '\n'.join(str(call[0][0]) for call in calls)
            
            assert 'Task Statistics' in output
            assert 'Total tasks: 2' in output
            assert 'Completion rate: 50.0%' in output
            assert 'COMPLETED: 1' in output
            assert 'PENDING: 1' in output
    
    def test_cmd_clear(self):
        """Test clear command."""
        self.cli.init_manager(self.temp_file)
        
        # Add and complete test tasks
        task_id1 = self.cli.manager.add_task('Task 1')
        task_id2 = self.cli.manager.add_task('Task 2')
        self.cli.manager.complete_task(task_id1)
        
        # Clear completed tasks with force
        args = MagicMock()
        args.force = True
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_clear(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert '✓' in output
            assert 'Cleared 1' in output
        
        # Verify only pending task remains
        tasks = self.cli.manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == task_id2
    
    def test_cmd_export(self):
        """Test export command."""
        self.cli.init_manager(self.temp_file)
        
        # Add test task
        self.cli.manager.add_task('Test Task')
        
        # Export tasks
        export_file = os.path.join(self.temp_dir, 'export.json')
        args = MagicMock()
        args.output_file = export_file
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_export(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert '✓' in output
            assert 'exported' in output.lower()
        
        # Verify export file was created
        assert os.path.exists(export_file)
        with open(export_file, 'r') as f:
            data = json.load(f)
            assert len(data) == 1
    
    def test_cmd_import(self):
        """Test import command."""
        self.cli.init_manager(self.temp_file)
        
        # Create test import file
        import_file = os.path.join(self.temp_dir, 'import.json')
        test_data = {
            'test-id': {
                'id': 'test-id',
                'title': 'Imported Task',
                'priority': 'HIGH',
                'status': 'PENDING',
                'created_at': '2023-01-01T12:00:00',
                'updated_at': None,
                'tags': [],
                'metadata': {}
            }
        }
        
        with open(import_file, 'w') as f:
            json.dump(test_data, f)
        
        # Import tasks
        args = MagicMock()
        args.input_file = import_file
        args.merge = False
        
        with patch('builtins.print') as mock_print:
            self.cli.cmd_import(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert '✓' in output
            assert 'imported' in output.lower()
        
        # Verify task was imported
        tasks = self.cli.manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == 'Imported Task'
    
    def test_cmd_import_file_not_found(self):
        """Test import command with non-existent file."""
        self.cli.init_manager(self.temp_file)
        
        args = MagicMock()
        args.input_file = 'non-existent.json'
        args.merge = False
        
        with patch('sys.stderr'), pytest.raises(SystemExit):
            self.cli.cmd_import(args)
    
    def test_run_no_args(self):
        """Test running CLI with no arguments."""
        # Capture stdout since parser.print_help() writes to stdout
        with patch('sys.stdout') as mock_stdout:
            self.cli.run([])
            # Should print help to stdout
            mock_stdout.write.assert_called()
    
    def test_run_help(self):
        """Test running CLI with help argument."""
        with patch('sys.stdout'):
            with pytest.raises(SystemExit):
                self.cli.run(['--help'])
    
    def test_run_add_command(self):
        """Test running CLI with add command."""
        args = ['-f', self.temp_file, 'add', 'Test Task', '-p', 'high']
        
        with patch('builtins.print') as mock_print:
            self.cli.run(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert '✓' in output
            assert 'Task added' in output
    
    def test_run_list_command(self):
        """Test running CLI with list command."""
        # First add a task
        self.cli.run(['-f', self.temp_file, 'add', 'Test Task'])
        
        # Then list tasks
        args = ['-f', self.temp_file, 'list']
        
        with patch('builtins.print') as mock_print:
            self.cli.run(args)
            mock_print.assert_called()
            output = str(mock_print.call_args[0][0])
            assert 'Test Task' in output
    
    def test_error_handling(self):
        """Test error handling."""
        with patch('sys.stderr') as mock_stderr:
            with pytest.raises(SystemExit) as excinfo:
                self.cli.error("Test error message")
            
            assert excinfo.value.code == 1
    
    def test_success_and_info_messages(self):
        """Test success and info message formatting."""
        with patch('builtins.print') as mock_print:
            self.cli.success("Test success")
            mock_print.assert_called_with("✓ Test success")
        
        with patch('builtins.print') as mock_print:
            self.cli.info("Test info")
            mock_print.assert_called_with("ℹ Test info")


class TestTodoCLIIntegration:
    """Integration tests for TodoCLI."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'integration_test.json')
        self.cli = TodoCLI()
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow(self):
        """Test complete workflow: add, list, complete, delete."""
        # Add tasks
        with patch('builtins.print'):
            self.cli.run(['-f', self.temp_file, 'add', 'Task 1', '-p', 'high', '-t', 'work'])
            self.cli.run(['-f', self.temp_file, 'add', 'Task 2', '-p', 'low', '-t', 'personal'])
        
        # List all tasks
        with patch('builtins.print') as mock_print:
            self.cli.run(['-f', self.temp_file, 'list'])
            output = str(mock_print.call_args[0][0])
            assert 'Task 1' in output
            assert 'Task 2' in output
            assert 'HIGH' in output
            assert 'LOW' in output
        
        # Get task IDs for further operations
        self.cli.init_manager(self.temp_file)
        tasks = self.cli.manager.list_tasks()
        task_id_1 = tasks[0].id  # High priority task should be first
        task_id_2 = tasks[1].id
        
        # Complete first task
        with patch('builtins.print') as mock_print:
            self.cli.run(['-f', self.temp_file, 'complete', task_id_1])
            output = str(mock_print.call_args[0][0])
            assert '✓' in output
            assert 'Completed 1 task' in output
        
        # List pending tasks only
        with patch('builtins.print') as mock_print:
            self.cli.run(['-f', self.temp_file, 'list', '-s', 'pending'])
            output = str(mock_print.call_args[0][0])
            assert 'Task 2' in output
            assert 'Task 1' not in output  # Should be filtered out
        
        # Delete second task
        with patch('builtins.print'):
            self.cli.run(['-f', self.temp_file, 'delete', task_id_2, '--force'])
        
        # Verify only completed task remains
        with patch('builtins.print') as mock_print:
            self.cli.run(['-f', self.temp_file, 'list'])
            output = str(mock_print.call_args[0][0])
            assert 'Task 1' in output
            assert 'Task 2' not in output
    
    def test_search_and_filter_workflow(self):
        """Test search and filtering functionality."""
        # Add diverse tasks
        with patch('builtins.print'):
            self.cli.run(['-f', self.temp_file, 'add', 'Buy groceries', '-t', 'shopping,food'])
            self.cli.run(['-f', self.temp_file, 'add', 'Grocery store visit', '-t', 'shopping'])
            self.cli.run(['-f', self.temp_file, 'add', 'Walk the dog', '-t', 'pets'])
        
        # Search for grocery-related tasks
        with patch('builtins.print') as mock_print:
            self.cli.run(['-f', self.temp_file, 'search', 'grocery'])
            # Get all call outputs and check them
            calls = mock_print.call_args_list
            all_output = ' '.join(str(call[0][0]) for call in calls)
            assert 'groceries' in all_output.lower() or 'grocery' in all_output.lower()
        
        # Filter by tag
        with patch('builtins.print') as mock_print:
            self.cli.run(['-f', self.temp_file, 'list', '-t', 'shopping'])
            output = str(mock_print.call_args[0][0])
            assert 'groceries' in output.lower() or 'grocery' in output.lower()
            assert 'dog' not in output.lower()
    
    def test_export_import_workflow(self):
        """Test export and import functionality."""
        # Add tasks
        with patch('builtins.print'):
            self.cli.run(['-f', self.temp_file, 'add', 'Task 1'])
            self.cli.run(['-f', self.temp_file, 'add', 'Task 2'])
        
        # Export tasks
        export_file = os.path.join(self.temp_dir, 'export.json')
        with patch('builtins.print'):
            self.cli.run(['-f', self.temp_file, 'export', export_file])
        
        # Create new todo file and import
        import_file = os.path.join(self.temp_dir, 'imported.json')
        with patch('builtins.print'):
            self.cli.run(['-f', import_file, 'import', export_file])
        
        # Verify imported tasks
        with patch('builtins.print') as mock_print:
            self.cli.run(['-f', import_file, 'list'])
            output = str(mock_print.call_args[0][0])
            assert 'Task 1' in output
            assert 'Task 2' in output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])