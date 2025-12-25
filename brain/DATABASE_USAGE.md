# CognitiveDatabase Usage Guide

## Overview

The `CognitiveDatabase` class centralizes all database operations that were previously scattered throughout the monolithic `CognitiveArchitecture` class. It provides a clean, focused API for database interactions while maintaining 100% backward compatibility.

## Key Benefits

- **Single Responsibility**: Handles only database operations
- **Improved Testability**: Can be mocked and tested in isolation  
- **Better Maintainability**: Database changes localized to one class
- **Enhanced Reusability**: Can be shared across cognitive components
- **Backward Compatibility**: Maintains existing schemas and APIs

## Basic Usage

### Initialization

```python
from brain.database import CognitiveDatabase, Reflection

# Use default path (~/.swarm/brain.db)
db = CognitiveDatabase()

# Or specify custom path
db = CognitiveDatabase("/path/to/custom/brain.db")
```

### Storing Reflections

```python
from datetime import datetime

reflection = Reflection(
    task="Implement new feature",
    outcome="Feature completed successfully", 
    success=True,
    what_worked=["Good planning", "Clear requirements"],
    what_failed=[],
    insights=["Planning reduces development time"],
    skill_candidates=[{"name": "feature_skill", "pattern": "implement feature", "tools": ["write", "test"]}],
    confidence=0.9,
    model_used="sonnet",
    tokens_used=2000,
    duration_seconds=45.0,
    created_at=datetime.now()
)

reflection_id = db.store_reflection(reflection)
```

### Querying Data

```python
# Get successful reflections for template matching
successful = db.get_successful_reflections(limit=10)

# Check for similar past solutions
has_similar = db.has_similar_past_solution("Implement another feature")

# Get insights for a task
insights = db.get_insights_for_task("Build new component")

# Get failure warnings
warnings = db.get_failure_warnings_for_task("Deploy to production")

# Get system statistics
stats = db.get_system_stats()
```

### Skill Management

```python
# Find existing similar skill
existing_skill = db.find_similar_skill("implement feature pattern")

if existing_skill:
    # Update existing skill
    db.update_skill_success_count(existing_skill["id"])
else:
    # Create new skill
    db.create_skill(
        skill_id="feature_skill_001",
        name="Feature Implementation Skill", 
        pattern="implement feature with tests",
        tools=["write", "test", "verify"]
    )
```

### Failure Pattern Tracking

```python
# Store failure pattern for learning
db.store_failure_pattern(
    task="Deploy application to production",
    error="Connection timeout after 30 seconds"
)

# Get failure patterns for analysis
patterns = db.get_failure_patterns(min_occurrences=2)
```

### Model Performance Analysis

```python
# Get performance stats by model
model_stats = db.get_model_performance_stats()

# Get recommended model based on performance
recommended = db.get_recommended_model("default_model")
```

## Advanced Usage

### Transaction Management

```python
# Use transactions for consistency
with db.transaction():
    db.store_reflection(reflection1)
    db.create_skill("skill_001", "Test Skill", "test pattern", ["test"])
    # Automatically commits on success, rolls back on error
```

### Direct Database Access

```python
# For complex queries not covered by the API
conn = db.get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT task, confidence 
    FROM reflections 
    WHERE success = 1 AND confidence > 0.8
    ORDER BY created_at DESC
""")

high_confidence_tasks = cursor.fetchall()
```

### Custom Analytics

```python
# Example: Get average performance by model
with db.transaction():
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT model_used,
               AVG(confidence) as avg_confidence,
               AVG(duration_seconds) as avg_duration
        FROM reflections 
        GROUP BY model_used
        HAVING COUNT(*) >= 5
    """)
    
    performance_data = [dict(row) for row in cursor.fetchall()]
```

## Database Schema

The CognitiveDatabase maintains the exact schema from the original brain.py:

### Tables
- **reflections**: Task outcomes and learnings
- **strategies**: Available cognitive strategies  
- **prompt_variants**: A/B testing data for prompts
- **skills**: Extracted reusable patterns
- **failure_patterns**: Tracked failure modes

### Schema Compatibility
- All table structures match original brain.py exactly
- Existing .swarm/brain.db files work without migration
- Column names and types preserved for backward compatibility

## Integration with Cognitive Components

### In CognitiveArchitecture
```python
class CognitiveArchitecture:
    def __init__(self, db_path=None):
        self.db = CognitiveDatabase(db_path)
        # Delegate all database operations to self.db
    
    def reflect(self, task, outcome, success, **kwargs):
        # Business logic here
        reflection = self._create_reflection(task, outcome, success, **kwargs)
        
        # Database operation delegated
        reflection_id = self.db.store_reflection(reflection)
        
        return reflection
```

### In Other Components
```python
class UncertaintyAssessor:
    def __init__(self, db: CognitiveDatabase):
        self.db = db
    
    def check_historical_failures(self, task):
        # Use shared database instance
        return self.db.get_failure_warnings_for_task(task)
```

## Testing

### Unit Testing
```python
import tempfile
from brain.database import CognitiveDatabase

def test_reflection_storage():
    with tempfile.NamedTemporaryFile(suffix='.db') as tmp_file:
        db = CognitiveDatabase(tmp_file.name)
        
        # Test database operations in isolation
        reflection_id = db.store_reflection(test_reflection)
        assert reflection_id is not None
        
        db.close()
```

### Mocking for Integration Tests
```python
from unittest.mock import Mock

def test_cognitive_architecture():
    mock_db = Mock(spec=CognitiveDatabase)
    mock_db.store_reflection.return_value = "test_id"
    
    brain = CognitiveArchitecture()
    brain.db = mock_db  # Inject mock
    
    brain.reflect("test task", "success", True)
    mock_db.store_reflection.assert_called_once()
```

## Migration from Original brain.py

The extraction maintains full backward compatibility:

### Before (brain.py)
```python
brain = get_brain()
brain.reflect(task, outcome, success, model, tokens, duration)
insights = brain.get_insights_for_task(task)
stats = brain.get_stats()
```

### After (with CognitiveDatabase)
```python
brain = get_brain()  # Still works exactly the same
brain.reflect(task, outcome, success, model, tokens, duration)
insights = brain.get_insights_for_task(task)  
stats = brain.get_stats()

# Plus new direct database access if needed
db_stats = brain.db.get_system_stats()  # New capability
```

## Connection Management

- Uses SQLite connection singleton pattern
- Automatic connection creation on first use
- Manual connection closing with `db.close()`
- Row factory enabled for dict-like result access
- Foreign keys enabled for referential integrity

## Error Handling

- Transactions automatically roll back on errors
- Connection errors bubble up for handling
- JSON serialization errors handled gracefully
- Database file creation handles permissions

## Performance Considerations

- Connection reuse minimizes overhead
- Prepared statements via parameterized queries  
- Indexes on frequently queried columns
- Transaction batching for bulk operations
- Connection closing for cleanup