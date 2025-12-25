# Cognitive Architecture Refactoring Plan

## Overview
Refactor the monolithic `CognitiveArchitecture` class (723 lines) into 6 specialized components while maintaining 100% backward compatibility with the existing public API.

## Phase 1: Create New Component Classes

### 1.1 CognitiveDatabase Class
**Purpose**: Centralized database management
**Extracted from**:
- `_init_db()` method
- All raw SQL queries
- Connection management

**Key Features**:
- Connection pooling (single connection for SQLite)
- Schema version management
- Query builders for common operations
- Transaction support

### 1.2 UncertaintyAssessor Class
**Purpose**: Task complexity and risk analysis
**Extracted from**:
- `assess_uncertainty()` method
- Complexity analysis logic
- Risk factor identification

**Dependencies**: CognitiveDatabase

### 1.3 StrategySelector Class
**Purpose**: Strategy selection and management
**Extracted from**:
- `_load_strategies()` method
- `select_strategy()` method
- `_has_similar_past_solution()` method
- Strategy matching logic

**Dependencies**: CognitiveDatabase, UncertaintyAssessor

### 1.4 ReflectionEngine Class
**Purpose**: Post-task analysis and learning
**Extracted from**:
- `reflect()` method
- `_store_reflection()` method
- Lesson extraction logic

**Dependencies**: CognitiveDatabase

### 1.5 SkillExtractor Class
**Purpose**: Pattern recognition and skill creation
**Extracted from**:
- `_maybe_create_skill()` method
- `_update_failure_pattern()` method
- Pattern matching logic

**Dependencies**: CognitiveDatabase, ReflectionEngine

### 1.6 KnowledgeRetriever Class
**Purpose**: Historical insights and recommendations
**Extracted from**:
- `get_insights_for_task()` method
- `get_failure_warnings()` method
- `get_recommended_model()` method

**Dependencies**: CognitiveDatabase

## Phase 2: Migration Strategy

### Step 1: Create Component Files
```
brain/
├── __init__.py
├── database.py      # CognitiveDatabase
├── uncertainty.py   # UncertaintyAssessor
├── strategy.py      # StrategySelector
├── reflection.py    # ReflectionEngine
├── skills.py        # SkillExtractor
├── knowledge.py     # KnowledgeRetriever
└── architecture.py  # Main CognitiveArchitecture
```

### Step 2: Implement Components
1. Start with CognitiveDatabase (no dependencies)
2. Implement UncertaintyAssessor
3. Implement ReflectionEngine
4. Implement StrategySelector
5. Implement SkillExtractor
6. Implement KnowledgeRetriever
7. Create new CognitiveArchitecture as orchestrator

### Step 3: Backward Compatibility Layer
The new `CognitiveArchitecture` class will:
- Maintain all existing public methods
- Delegate to appropriate components
- Preserve method signatures
- Return same data structures

### Step 4: Integration Testing
1. Create comprehensive test suite for new components
2. Verify backward compatibility with existing tests
3. Performance benchmarking
4. Integration testing with dependent modules

## Phase 3: Gradual Migration

### Stage 1: Parallel Implementation
- Keep existing brain.py intact
- Implement new architecture in brain/ directory
- Add feature flag for switching between implementations

### Stage 2: Testing and Validation
- Run both implementations in parallel
- Compare outputs for consistency
- Monitor performance metrics
- Gather feedback from integration points

### Stage 3: Switchover
- Update imports to use new architecture
- Deprecate old brain.py
- Provide migration guide for any custom extensions

## Backward Compatibility Checklist

### Public API Methods (Must Preserve)
- [x] `assess_uncertainty(task: str, context: str) -> Uncertainty`
- [x] `select_strategy(task: str, context: str, uncertainty: Optional[Uncertainty]) -> Strategy`
- [x] `reflect(task: str, result: Any, context: Dict[str, Any]) -> Reflection`
- [x] `get_insights_for_task(task: str, limit: int) -> List[str]`
- [x] `get_failure_warnings(task: str, approach: str) -> List[str]`
- [x] `get_recommended_model(task: str, complexity: str) -> str`
- [x] `get_stats() -> Dict[str, Any]`
- [x] `shutdown() -> None`

### Data Structures (Must Preserve)
- [x] `Reflection` dataclass
- [x] `Strategy` dataclass  
- [x] `Uncertainty` dataclass

### Global Functions (Must Preserve)
- [x] `get_brain(base_path: str) -> CognitiveArchitecture`

### Database Schema (Must Preserve)
- [x] reflections table
- [x] strategies table
- [x] prompt_variants table
- [x] skills table
- [x] failure_patterns table

## Benefits of Refactoring

### Immediate Benefits
1. **Modularity**: Each component can be developed and tested independently
2. **Clarity**: Clear separation of concerns
3. **Testability**: Easier to write focused unit tests
4. **Maintainability**: Changes isolated to specific components

### Future Benefits
1. **Extensibility**: New cognitive functions can be added as new components
2. **Performance**: Components can be optimized independently
3. **Scalability**: Components could be distributed if needed
4. **Reusability**: Components can be used in other contexts

## Risk Mitigation

### Risks
1. **Breaking Changes**: Existing code depends on current implementation
2. **Performance Regression**: Multiple objects vs single class
3. **Data Consistency**: Multiple components accessing same database
4. **Integration Complexity**: More moving parts

### Mitigation Strategies
1. **Comprehensive Testing**: Full test coverage before switchover
2. **Performance Monitoring**: Benchmark key operations
3. **Database Transactions**: Ensure ACID properties
4. **Gradual Rollout**: Feature flags for safe deployment

## Success Criteria
1. All existing tests pass without modification
2. No performance degradation (< 5% overhead)
3. Zero breaking changes to public API
4. Improved code clarity and maintainability
5. Easier to add new cognitive capabilities

## Timeline Estimate
- Phase 1 (Component Creation): 2-3 days
- Phase 2 (Implementation): 3-4 days  
- Phase 3 (Migration): 2-3 days
- Testing & Validation: 2-3 days

**Total: 9-13 days** for complete refactoring with full backward compatibility