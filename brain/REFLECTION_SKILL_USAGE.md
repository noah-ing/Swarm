# ReflectionEngine and SkillExtractor Usage Guide

## Overview

The **ReflectionEngine** and **SkillExtractor** are specialized cognitive components extracted from the monolithic `CognitiveArchitecture` class. They handle post-task learning and pattern recognition for building autonomous cognitive capabilities.

## Components

### ReflectionEngine
- **Purpose**: Post-task analysis and learning extraction
- **Responsibilities**: 
  - Analyze task outcomes for insights
  - Calculate confidence scores
  - Coordinate with SkillExtractor
  - Provide recommendations based on history

### SkillExtractor  
- **Purpose**: Pattern recognition and skill creation
- **Responsibilities**:
  - Identify reusable patterns from successful tasks
  - Create and validate skills
  - Track failure patterns
  - Manage skill lifecycle

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  ReflectionEngine │◄──►│ CognitiveDatabase │◄──►│  SkillExtractor │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌──────────────────┐
                    │   SQLite Brain   │
                    │    Database      │
                    └──────────────────┘
```

## Usage Examples

### Basic Setup

```python
from brain.database import CognitiveDatabase
from brain.reflection import ReflectionEngine  
from brain.skill_extractor import SkillExtractor

# Initialize components
db = CognitiveDatabase()
skill_extractor = SkillExtractor(db)
reflection_engine = ReflectionEngine(db, skill_extractor)
```

### Conducting Reflection

```python
# After a successful task
reflection = reflection_engine.reflect(
    task="Implement Python function for data processing",
    outcome="Successfully created function with unit tests",
    success=True,
    model_used="claude-3.5-sonnet",
    tokens_used=2500,
    duration_seconds=8.5,
    tool_calls=[
        {"name": "write", "parameters": {"file_path": "processor.py"}},
        {"name": "bash", "parameters": {"command": "python -m pytest"}}
    ]
)

print(f"Confidence: {reflection.confidence}")
print(f"Insights: {reflection.insights}")
```

### Skill Extraction

```python
# Skills are automatically created during reflection if candidates are found
# You can also manually extract skills from reflection data

reflection_data = {
    "skill_candidates": [
        {
            "name": "python_test_pattern",
            "pattern": "Create Python module with comprehensive unit tests",
            "tools": ["write", "bash"]
        }
    ]
}

skill_ids = skill_extractor.extract_skills_from_reflection(reflection_data)
print(f"Created skills: {skill_ids}")
```

### Getting Insights and Warnings

```python
# Get insights for similar tasks
insights = reflection_engine.get_insights_for_task("Python data processing")

# Get failure warnings
warnings = reflection_engine.get_failure_warnings("Deploy to production")

# Get model recommendations
recommended_model = reflection_engine.get_recommended_model("Python coding task")
```

## Skill Creation Criteria

Skills are created when patterns meet these criteria:

### Valid Skill Candidate
- ✅ Pattern length: 20-500 characters
- ✅ Tool sequence: 1-10 tools
- ✅ Not overly generic (e.g., "fix", "update")
- ✅ Substantial enough for reuse

### Skill Creation Decision
- ✅ Pattern length: 30+ characters
- ✅ Contains valuable tool patterns
- ✅ Complexity score: 0.3-0.8
- ✅ Specific enough to be useful

### Valuable Tool Patterns
- File operations: `["read", "write"]`, `["search", "read", "write"]`
- Development workflows: `["read", "write", "bash"]`
- Multi-step analysis: `["search", "read", "search"]`

## Database Integration

Both components use the **CognitiveDatabase** for persistence:

### Tables Used
- **reflections**: Stores all reflection data
- **skills**: Stores extracted skill patterns  
- **failure_patterns**: Tracks recurring failures

### Shared Operations
- Reflection storage and retrieval
- Skill creation and updates
- Failure pattern tracking
- Analytics and statistics

## Configuration

### ReflectionEngine Configuration

```python
# Confidence score factors
BASE_CONFIDENCE_SUCCESS = 0.7
BASE_CONFIDENCE_FAILURE = 0.5
INSIGHT_BONUS = 0.1
SKILL_CANDIDATE_BONUS = 0.1
EFFICIENT_TOKEN_BONUS = 0.05

# Task complexity estimation
COMPLEX_OPERATIONS = ["refactor", "architect", "design", "implement"]
TECHNICAL_TERMS = ["database", "api", "algorithm", "pattern"]
```

### SkillExtractor Configuration

```python
# Validation thresholds
MIN_PATTERN_LENGTH = 20
MAX_PATTERN_LENGTH = 500  
MAX_TOOL_SEQUENCE = 10
MIN_SKILL_PATTERN_LENGTH = 30

# Complexity scoring
MIN_COMPLEXITY = 0.3
MAX_COMPLEXITY = 0.8
```

## Performance Considerations

### ReflectionEngine Performance
- **O(1)** for basic reflection creation
- **O(n)** for insight retrieval (limited to recent items)
- **Efficient** token usage analysis
- **Fast** confidence calculation

### SkillExtractor Performance  
- **O(1)** for skill validation
- **O(n)** for pattern matching (with database limits)
- **Efficient** tool pattern analysis
- **Smart** duplicate detection

## Error Handling

### Common Error Patterns Tracked
- **Timeout errors**: Suggest task decomposition
- **Token limits**: Recommend context reduction
- **Not found errors**: Verify resources first
- **Permission errors**: Check access rights
- **Syntax errors**: Validate before execution

### Failure Pattern Analysis
```python
# Automatic categorization of errors
ERROR_TYPES = {
    "timeout": "Task took too long",
    "token_limit": "Context too large", 
    "not_found": "Resource missing",
    "permission": "Access denied",
    "syntax": "Code validation failed"
}
```

## Monitoring and Analytics

### Reflection Stats
```python
stats = reflection_engine.get_reflection_stats()
# Returns: total_reflections, successful_reflections, success_rate, avg_confidence
```

### Skill Extraction Stats  
```python
stats = skill_extractor.get_skill_extraction_stats()
# Returns: total_skills, failure_patterns, skill_extraction_rate
```

### Database Health
```python
health = db.get_system_stats()
# Returns comprehensive system statistics
```

## Best Practices

### 1. Dependency Injection
Always inject `SkillExtractor` into `ReflectionEngine` for proper integration:

```python
# ✅ Correct
reflection_engine = ReflectionEngine(db, skill_extractor)

# ❌ Missing integration
reflection_engine = ReflectionEngine(db)  # Skills won't be created automatically
```

### 2. Quality Skill Patterns
Focus on creating substantial, reusable patterns:

```python
# ✅ Good skill candidate
{
    "pattern": "Analyze Python codebase by reading files, searching for patterns, and refactoring for better structure",
    "tools": ["read", "search", "write", "bash"]
}

# ❌ Poor skill candidate  
{
    "pattern": "fix bug",
    "tools": ["write"]
}
```

### 3. Comprehensive Reflections
Provide detailed context for better learning:

```python
# ✅ Detailed reflection
reflection_engine.reflect(
    task="Detailed task description with context",
    outcome="Specific outcome with metrics",
    success=True,
    model_used="claude-3.5-sonnet",
    tokens_used=2500,
    duration_seconds=8.5,
    tool_calls=detailed_tool_calls  # Include all tool usage
)
```

### 4. Regular Monitoring
Monitor system health regularly:

```python
# Check reflection quality
stats = reflection_engine.get_reflection_stats()
if stats['avg_confidence'] < 0.6:
    print("Warning: Low reflection confidence")

# Check skill extraction rate
skill_stats = skill_extractor.get_skill_extraction_stats()
if skill_stats['skill_extraction_rate'] < 0.1:
    print("Info: Consider adjusting skill criteria")
```

## Integration with Existing Systems

### With CognitiveArchitecture
The components integrate seamlessly with the existing brain system:

```python
from brain import get_brain

brain = get_brain()
# Brain internally uses ReflectionEngine and SkillExtractor
reflection = brain.reflect(...)  # Uses new components
```

### With Background Learning
```python
from background import get_background_learner

learner = get_background_learner()
# Background learner can use reflection data for continuous improvement
```

### With Memory System
```python
from memory import get_memory_store

memory = get_memory_store()  
# Memory can leverage skills for task optimization
```

## Troubleshooting

### Skills Not Being Created
1. **Check pattern length**: Must be 20-500 characters
2. **Verify tool sequence**: Must have 1-10 tools  
3. **Review complexity**: Pattern must be substantial but not overly complex
4. **Check integration**: Ensure SkillExtractor is injected into ReflectionEngine

### Low Reflection Confidence
1. **Provide more context**: Include detailed task descriptions
2. **Add tool calls**: Include comprehensive tool usage data
3. **Check token usage**: Ensure reasonable token consumption
4. **Verify success indicators**: Confirm task success/failure is accurate

### Performance Issues
1. **Limit data retrieval**: Use reasonable limits for insight queries
2. **Monitor database size**: Regular cleanup of old reflections
3. **Check complexity calculations**: Ensure efficient pattern analysis
4. **Optimize skill validation**: Review validation criteria

This completes the comprehensive usage guide for the ReflectionEngine and SkillExtractor components.