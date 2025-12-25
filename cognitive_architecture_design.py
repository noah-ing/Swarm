"""
Cognitive Architecture Refactoring Design
=========================================

This module defines the interfaces for the new cognitive architecture
with 6 separate classes that maintain backward compatibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import sqlite3
from enum import Enum


# Existing data structures (must be preserved for compatibility)
@dataclass
class Reflection:
    task_description: str
    context: str
    approach: str
    outcome: str
    lessons_learned: List[str]
    timestamp: str
    model_used: str
    uncertainty_level: str
    task_complexity: str
    was_successful: bool


@dataclass
class Strategy:
    name: str
    description: str
    when_to_use: List[str]
    steps: List[str]
    requires_planning: bool
    supports_delegation: bool
    model_flexibility: str


@dataclass
class Uncertainty:
    level: str  # 'low', 'medium', 'high'
    score: float  # 0.0 to 1.0
    reasons: List[str]
    suggested_actions: List[str]


class LearningMode(Enum):
    TARGETED = "targeted"
    EXPLORATORY = "exploratory"
    REFLECTIVE = "reflective"


# New component interfaces
class CognitiveDatabase:
    """
    Centralized database management for all cognitive components.
    Handles connection pooling, schema management, and data access.
    """
    
    def __init__(self, base_path: str = ".swarm"):
        self.base_path = base_path
        self.db_path = f"{base_path}/brain.db"
        self._connection = None
        
    def get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        pass
        
    def init_schema(self) -> None:
        """Initialize all required database tables."""
        pass
        
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts."""
        pass
        
    def execute_write(self, query: str, params: Optional[Tuple] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        pass
        
    def close(self) -> None:
        """Close database connection."""
        pass


class UncertaintyAssessor:
    """
    Analyzes task complexity and assesses uncertainty/risk factors.
    """
    
    def __init__(self, db: CognitiveDatabase):
        self.db = db
        self._complexity_indicators = {
            'high': ['complex', 'integrate', 'architect', 'refactor', 'optimize'],
            'medium': ['implement', 'create', 'update', 'modify', 'analyze'],
            'low': ['fix', 'add', 'remove', 'rename', 'move']
        }
        
    def assess_uncertainty(self, task: str, context: str) -> Uncertainty:
        """
        Assess the uncertainty level of a task.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Uncertainty object with level, score, reasons, and suggested actions
        """
        pass
        
    def analyze_complexity(self, task: str) -> str:
        """
        Determine task complexity: 'low', 'medium', or 'high'.
        """
        pass
        
    def get_risk_factors(self, task: str, context: str) -> List[str]:
        """
        Identify specific risk factors for the task.
        """
        pass
        
    def suggest_mitigation_actions(self, risk_factors: List[str]) -> List[str]:
        """
        Suggest actions to mitigate identified risks.
        """
        pass


class StrategySelector:
    """
    Selects appropriate strategies for task execution.
    """
    
    def __init__(self, db: CognitiveDatabase, uncertainty_assessor: UncertaintyAssessor):
        self.db = db
        self.uncertainty_assessor = uncertainty_assessor
        self.strategies: Dict[str, Strategy] = {}
        self._load_strategies()
        
    def _load_strategies(self) -> None:
        """Load built-in and learned strategies."""
        pass
        
    def select_strategy(self, task: str, context: str, 
                       uncertainty: Optional[Uncertainty] = None) -> Strategy:
        """
        Select the most appropriate strategy for a task.
        
        Args:
            task: Task description
            context: Additional context
            uncertainty: Pre-assessed uncertainty (optional)
            
        Returns:
            Selected Strategy object
        """
        pass
        
    def add_strategy(self, strategy: Strategy) -> None:
        """Add a new strategy to the selector."""
        pass
        
    def get_strategy(self, name: str) -> Optional[Strategy]:
        """Get a specific strategy by name."""
        pass
        
    def list_strategies(self) -> List[Strategy]:
        """List all available strategies."""
        pass
        
    def find_similar_past_solution(self, task: str) -> Optional[Dict[str, Any]]:
        """Find similar successfully completed tasks."""
        pass


class ReflectionEngine:
    """
    Handles post-task reflection and learning extraction.
    """
    
    def __init__(self, db: CognitiveDatabase):
        self.db = db
        
    def reflect(self, task: str, result: Any, context: Dict[str, Any]) -> Reflection:
        """
        Analyze task execution and extract learnings.
        
        Args:
            task: Original task description
            result: Task execution result
            context: Execution context (approach, model, timing, etc.)
            
        Returns:
            Reflection object with analysis and lessons
        """
        pass
        
    def store_reflection(self, reflection: Reflection) -> None:
        """Store reflection in database."""
        pass
        
    def get_reflections(self, limit: int = 10, 
                       task_pattern: Optional[str] = None) -> List[Reflection]:
        """Retrieve past reflections."""
        pass
        
    def extract_lessons(self, task: str, outcome: str, 
                       approach: str, was_successful: bool) -> List[str]:
        """Extract specific lessons from task execution."""
        pass
        
    def identify_patterns(self, reflections: List[Reflection]) -> Dict[str, List[str]]:
        """Identify patterns across multiple reflections."""
        pass


class SkillExtractor:
    """
    Recognizes patterns and extracts reusable skills from experiences.
    """
    
    def __init__(self, db: CognitiveDatabase, reflection_engine: ReflectionEngine):
        self.db = db
        self.reflection_engine = reflection_engine
        
    def maybe_create_skill(self, reflection: Reflection) -> Optional[Dict[str, Any]]:
        """
        Analyze reflection to potentially create a new skill.
        
        Args:
            reflection: Completed task reflection
            
        Returns:
            Skill dict if pattern detected, None otherwise
        """
        pass
        
    def extract_skill_pattern(self, task: str, approach: str, 
                            outcome: str) -> Optional[str]:
        """Extract reusable pattern from successful execution."""
        pass
        
    def store_skill(self, skill: Dict[str, Any]) -> None:
        """Store extracted skill in database."""
        pass
        
    def get_skills(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve stored skills."""
        pass
        
    def match_skill_to_task(self, task: str) -> Optional[Dict[str, Any]]:
        """Find applicable skill for given task."""
        pass
        
    def update_failure_pattern(self, task: str, approach: str, 
                             error: str, context: str) -> None:
        """Track failure patterns for future avoidance."""
        pass


class KnowledgeRetriever:
    """
    Retrieves contextual knowledge and insights from past experiences.
    """
    
    def __init__(self, db: CognitiveDatabase):
        self.db = db
        
    def get_insights_for_task(self, task: str, limit: int = 5) -> List[str]:
        """
        Get relevant insights from past experiences.
        
        Args:
            task: Current task description
            limit: Maximum insights to return
            
        Returns:
            List of relevant insights
        """
        pass
        
    def get_failure_warnings(self, task: str, approach: str) -> List[str]:
        """Get warnings based on past failures."""
        pass
        
    def get_recommended_model(self, task: str, 
                            complexity: str = "medium") -> str:
        """Recommend AI model based on past performance."""
        pass
        
    def get_success_patterns(self, task_type: str) -> List[Dict[str, Any]]:
        """Retrieve successful execution patterns."""
        pass
        
    def search_knowledge(self, query: str, 
                        knowledge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search knowledge base with flexible query."""
        pass
        
    def get_contextual_tips(self, task: str, strategy: str) -> List[str]:
        """Get tips specific to task and strategy combination."""
        pass


class CognitiveArchitecture:
    """
    Main cognitive architecture that orchestrates all components.
    Maintains backward compatibility with existing API.
    """
    
    def __init__(self, base_path: str = ".swarm"):
        self.base_path = base_path
        
        # Initialize components
        self.db = CognitiveDatabase(base_path)
        self.db.init_schema()
        
        self.uncertainty_assessor = UncertaintyAssessor(self.db)
        self.strategy_selector = StrategySelector(self.db, self.uncertainty_assessor)
        self.reflection_engine = ReflectionEngine(self.db)
        self.skill_extractor = SkillExtractor(self.db, self.reflection_engine)
        self.knowledge_retriever = KnowledgeRetriever(self.db)
        
    # Backward compatibility methods (existing public API)
    def assess_uncertainty(self, task: str, context: str = "") -> Uncertainty:
        """Assess uncertainty of a task (backward compatible)."""
        return self.uncertainty_assessor.assess_uncertainty(task, context)
        
    def select_strategy(self, task: str, context: str = "",
                       uncertainty: Optional[Uncertainty] = None) -> Strategy:
        """Select strategy for task (backward compatible)."""
        return self.strategy_selector.select_strategy(task, context, uncertainty)
        
    def reflect(self, task: str, result: Any, context: Dict[str, Any]) -> Reflection:
        """Reflect on task execution (backward compatible)."""
        reflection = self.reflection_engine.reflect(task, result, context)
        
        # Also trigger skill extraction
        self.skill_extractor.maybe_create_skill(reflection)
        
        return reflection
        
    def get_insights_for_task(self, task: str, limit: int = 5) -> List[str]:
        """Get insights for task (backward compatible)."""
        return self.knowledge_retriever.get_insights_for_task(task, limit)
        
    def get_failure_warnings(self, task: str, approach: str = "") -> List[str]:
        """Get failure warnings (backward compatible)."""
        return self.knowledge_retriever.get_failure_warnings(task, approach)
        
    def get_recommended_model(self, task: str, complexity: str = "medium") -> str:
        """Get recommended model (backward compatible)."""
        return self.knowledge_retriever.get_recommended_model(task, complexity)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cognitive system statistics (backward compatible)."""
        stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM reflections) as total_reflections,
            (SELECT COUNT(*) FROM reflections WHERE was_successful = 1) as successful_tasks,
            (SELECT COUNT(*) FROM strategies) as total_strategies,
            (SELECT COUNT(*) FROM skills) as total_skills,
            (SELECT COUNT(*) FROM failure_patterns) as known_failure_patterns
        """
        results = self.db.execute_query(stats_query)
        return results[0] if results else {}
        
    def shutdown(self) -> None:
        """Cleanup resources (backward compatible)."""
        self.db.close()


# Global singleton management (backward compatible)
_brain_instance: Optional[CognitiveArchitecture] = None


def get_brain(base_path: str = ".swarm") -> CognitiveArchitecture:
    """Get or create the global brain instance (backward compatible)."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = CognitiveArchitecture(base_path)
    return _brain_instance