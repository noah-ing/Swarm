# UncertaintyAssessor and StrategySelector Usage Guide

This guide demonstrates how to use the extracted `UncertaintyAssessor` and `StrategySelector` components that have been extracted from the monolithic `CognitiveArchitecture` class.

## Overview

These components provide focused cognitive capabilities:

- **UncertaintyAssessor**: Analyzes task complexity and risk factors
- **StrategySelector**: Manages strategy selection and loading based on task characteristics

## Quick Start

```python
from brain import CognitiveDatabase, UncertaintyAssessor, StrategySelector

# Initialize components
db = CognitiveDatabase()  # Uses default ~/.swarm/brain.db path
assessor = UncertaintyAssessor(db)
selector = StrategySelector(db)

# Assess uncertainty for a task
uncertainty = assessor.assess_uncertainty("Refactor authentication system")
print(f"Uncertainty: {uncertainty.level} ({uncertainty.score:.3f})")
print(f"Action: {uncertainty.recommended_action}")

# Select strategy based on task
strategy = selector.select_strategy("Refactor authentication system")
print(f"Strategy: {strategy.name}")
print(f"Steps: {strategy.steps}")
```

## UncertaintyAssessor

### Core Functionality

The `UncertaintyAssessor` evaluates tasks across multiple dimensions:

#### 1. Basic Usage

```python
from brain.uncertainty import UncertaintyAssessor, Uncertainty
from brain.database import CognitiveDatabase

db = CognitiveDatabase()
assessor = UncertaintyAssessor(db)

# Simple task assessment
task = "Fix typo in README.md"
uncertainty = assessor.assess_uncertainty(task)

print(f"Level: {uncertainty.level}")           # low, medium, high, critical
print(f"Score: {uncertainty.score}")           # 0.0 to 1.0
print(f"Reasons: {uncertainty.reasons}")       # List of reasons
print(f"Action: {uncertainty.recommended_action}")  # proceed, cautious, escalate, ask_user
print(f"Fallbacks: {uncertainty.fallback_strategies}")  # Fallback strategies
```

#### 2. Risk Analysis

Get detailed risk breakdown:

```python
task = "Delete all production database records"
risk_analysis = assessor.get_risk_analysis(task)

for category, risks in risk_analysis.items():
    if risks:
        print(f"{category}: {risks}")

# Output:
# critical_operations: ['delete']
# production_risks: ['production']
# high_risk_operations: ['database']
```

#### 3. Complexity Analysis

Understand task complexity indicators:

```python
task = "Implement OAuth2 authentication with security validation"
complexity = assessor.get_complexity_indicators(task)

for level, indicators in complexity.items():
    if indicators:
        print(f"{level} complexity: {indicators}")

# Output:
# high complexity: ['authentication', 'security']
# medium complexity: ['implement']
```

### Uncertainty Levels

| Level | Score Range | Action | Description |
|-------|-------------|--------|-------------|
| `low` | 0.0 - 0.2 | `proceed` | Task is straightforward, low risk |
| `medium` | 0.2 - 0.4 | `cautious` | Some complexity/risk factors present |
| `high` | 0.4 - 0.7 | `escalate` | Significant risk/complexity requires care |
| `critical` | 0.7 - 1.0 | `ask_user` | High-risk operation requiring confirmation |

### Risk Factors Detected

The assessor identifies multiple risk categories:

- **Vague Language**: "somehow", "maybe", "something"
- **Unclear Requirements**: "not sure", "don't know"
- **Critical Operations**: "delete", "remove all", "truncate"
- **Production Context**: "production", "live", "prod"
- **Architectural Complexity**: "refactor entire", "system-wide"
- **High Complexity Domains**: "security", "authentication", "performance"
- **Historical Failures**: Similar past task failures

## StrategySelector

### Core Functionality

The `StrategySelector` manages strategy selection based on task patterns and uncertainty:

#### 1. Basic Usage

```python
from brain.strategy import StrategySelector, Strategy
from brain.database import CognitiveDatabase

db = CognitiveDatabase()
selector = StrategySelector(db)

# Select strategy for a task
task = "Show all Python files in the project"
strategy = selector.select_strategy(task)

print(f"Strategy: {strategy.name}")
print(f"Description: {strategy.description}")
print(f"Steps: {strategy.steps}")
print(f"Success Rate: {strategy.success_rate}")
```

#### 2. Available Strategies

The selector provides 6 built-in strategies:

```python
all_strategies = selector.get_all_strategies()
for name, strategy in all_strategies.items():
    print(f"{name}: {strategy.description}")

# Output:
# direct_execution: Run task directly with single agent
# decompose_parallel: Split into independent subtasks, run in parallel
# decompose_sequential: Split into dependent subtasks, run in order
# explore_first: Understand before acting
# trial_and_error: Try approaches, learn from failures
# template_match: Find similar past solution, adapt it
```

#### 3. Strategy Recommendations

Get detailed recommendations with rationale:

```python
task = "Refactor authentication system with OAuth2"
recommendations = selector.get_strategy_recommendations(task)

print(f"Primary: {recommendations['primary_strategy'].name}")
print(f"Uncertainty: {recommendations['uncertainty_assessment'].level}")
print(f"Alternatives: {[s.name for s in recommendations['alternative_strategies']]}")
print(f"Rationale: {recommendations['selection_rationale']}")

# Output:
# Primary: Explore First
# Uncertainty: high
# Alternatives: ['Sequential Decomposition', 'Template Match']
# Rationale: Selected Explore First: High uncertainty (high) suggests exploratory approach
```

### Strategy Selection Logic

The selector uses a priority-based approach:

1. **Uncertainty Check**: High/critical uncertainty → `Explore First`
2. **Template Matching**: Similar past solution → `Template Match`
3. **Sequential Patterns**: "first", "then", "finally" → `Sequential Decomposition`
4. **Parallel Operations**: Independent "and" operations → `Parallel Decomposition`
5. **Simple Queries**: "show", "list", "find" → `Direct Execution`
6. **Single File Ops**: Simple file operations → `Direct Execution`
7. **Complex Tasks**: Architecture/design tasks → `Explore First` or `Sequential Decomposition`

### Performance Tracking

Update strategy performance based on outcomes:

```python
# After executing a task with a strategy
selector.update_strategy_performance("direct_execution", success=True)

# Check updated statistics
strategy = selector.get_strategy_by_name("direct_execution")
print(f"Uses: {strategy.uses}")
print(f"Success Rate: {strategy.success_rate:.3f}")
```

## Integration Between Components

### Uncertainty-Driven Strategy Selection

The components work together for intelligent strategy selection:

```python
from brain import CognitiveDatabase, UncertaintyAssessor, StrategySelector

db = CognitiveDatabase()
assessor = UncertaintyAssessor(db)
selector = StrategySelector(db)

task = "Delete production database and rebuild schema"

# Assess uncertainty first
uncertainty = assessor.assess_uncertainty(task)
print(f"Uncertainty: {uncertainty.level} ({uncertainty.score:.3f})")

# Select strategy based on uncertainty
strategy = selector.select_strategy(task, uncertainty=uncertainty)
print(f"Strategy: {strategy.name}")

if uncertainty.level == "critical":
    print("⚠️ Critical operation detected - requires user confirmation")
    print(f"Fallback strategies: {uncertainty.fallback_strategies}")
```

### Historical Learning

Both components learn from historical data:

```python
# UncertaintyAssessor uses failure patterns
warnings = db.get_failure_warnings_for_task(task)
if warnings:
    print("Historical warnings:")
    for warning in warnings:
        print(f"  - {warning}")

# StrategySelector uses successful reflections for template matching
has_template = selector._has_similar_past_solution(task)
if has_template:
    print("Similar past solution found - using template matching")
```

## Advanced Usage

### Custom Risk Assessment

Extend the uncertainty assessor for domain-specific risks:

```python
class CustomUncertaintyAssessor(UncertaintyAssessor):
    def assess_uncertainty(self, task: str, context: str = "") -> Uncertainty:
        # Get base assessment
        uncertainty = super().assess_uncertainty(task, context)
        
        # Add custom domain logic
        if "machine learning" in task.lower():
            uncertainty.score += 0.2
            uncertainty.reasons.append("ML tasks have high uncertainty")
        
        # Recalculate level if needed
        if uncertainty.score >= 0.7:
            uncertainty.level = "critical"
        elif uncertainty.score >= 0.4:
            uncertainty.level = "high"
        
        return uncertainty
```

### Strategy Pattern Matching

Find strategies for specific patterns:

```python
# Find strategies applicable to a specific pattern
applicable = selector.get_strategies_for_pattern("multiple files")
for strategy in applicable:
    print(f"{strategy.name} - Success Rate: {strategy.success_rate:.3f}")
```

### Database Integration

Both components share the same database for consistency:

```python
# Same database instance ensures consistent data access
db = CognitiveDatabase()

# Components share historical data
assessor = UncertaintyAssessor(db)
selector = StrategySelector(db)

# Both can access failure patterns, reflections, etc.
stats = db.get_system_stats()
print(f"Total reflections: {stats['total_reflections']}")
print(f"Success rate: {stats['successful_reflections'] / stats['total_reflections']:.3f}")
```

## Benefits of Extraction

### Separation of Concerns
- **UncertaintyAssessor**: Focused on risk analysis and complexity evaluation
- **StrategySelector**: Focused on strategy management and selection logic
- **CognitiveDatabase**: Centralized data management

### Testability
```python
# Components can be tested independently
def test_high_risk_detection():
    db = CognitiveDatabase(":memory:")  # In-memory for testing
    assessor = UncertaintyAssessor(db)
    
    result = assessor.assess_uncertainty("Delete all production data")
    assert result.level == "critical"
    assert "CRITICAL" in result.reasons[0]
```

### Maintainability
- Each component has a single responsibility
- Easy to modify uncertainty logic without affecting strategy selection
- Clear interfaces between components

### Extensibility
- New uncertainty factors can be added to `UncertaintyAssessor`
- New strategies can be added to `StrategySelector`
- Components can be composed differently for different use cases

## Migration from CognitiveArchitecture

The extracted components maintain the exact same API as the original methods:

```python
# Old way (still works for backward compatibility)
from brain import get_brain
brain = get_brain()
uncertainty = brain.assess_uncertainty("task")
strategy = brain.select_strategy("task")

# New way (preferred for new code)
from brain import CognitiveDatabase, UncertaintyAssessor, StrategySelector
db = CognitiveDatabase()
assessor = UncertaintyAssessor(db)
selector = StrategySelector(db)
uncertainty = assessor.assess_uncertainty("task")
strategy = selector.select_strategy("task")
```

Both approaches produce identical results, ensuring seamless migration.