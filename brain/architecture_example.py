"""
Example: Refactored Cognitive Architecture with Backward Compatibility
=====================================================================

This demonstrates how the new modular architecture maintains 100% backward
compatibility with the existing brain.py API while providing better
separation of concerns.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import new components
from brain.database import CognitiveDatabase
from brain.uncertainty import UncertaintyAssessor, Uncertainty
# Note: Other components would be imported similarly:
# from brain.strategy import StrategySelector, Strategy
# from brain.reflection import ReflectionEngine, Reflection
# from brain.skills import SkillExtractor
# from brain.knowledge import KnowledgeRetriever


# Example stub for Strategy (would be in brain/strategy.py)
@dataclass
class Strategy:
    name: str
    description: str
    when_to_use: List[str]
    steps: List[str]
    requires_planning: bool
    supports_delegation: bool
    model_flexibility: str


# Example stub for Reflection (would be in brain/reflection.py)
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


class CognitiveArchitecture:
    """
    Main cognitive architecture that orchestrates all components.
    
    This class maintains 100% backward compatibility with the existing
    brain.py API while delegating to specialized components internally.
    
    Key features:
    - Same public methods as original CognitiveArchitecture
    - Same return types and behaviors
    - Improved internal organization
    - Better testability and maintainability
    """
    
    def __init__(self, base_path: str = ".swarm"):
        """Initialize cognitive architecture with all components."""
        self.base_path = base_path
        
        # Initialize components in dependency order
        self.db = CognitiveDatabase(base_path)
        self.db.init_schema()
        
        self.uncertainty_assessor = UncertaintyAssessor(self.db)
        
        # These would be initialized with actual implementations:
        # self.strategy_selector = StrategySelector(self.db, self.uncertainty_assessor)
        # self.reflection_engine = ReflectionEngine(self.db)
        # self.skill_extractor = SkillExtractor(self.db, self.reflection_engine)
        # self.knowledge_retriever = KnowledgeRetriever(self.db)
        
        # For now, we'll show the pattern with stubs
        self._strategies = self._load_default_strategies()
        
    # ===== BACKWARD COMPATIBILITY API =====
    # These are the exact same public methods as the original brain.py
    
    def assess_uncertainty(self, task: str, context: str = "") -> Uncertainty:
        """
        Assess uncertainty of a task.
        
        BACKWARD COMPATIBLE: Same signature and return type as original.
        Delegates to UncertaintyAssessor component.
        """
        return self.uncertainty_assessor.assess_uncertainty(task, context)
        
    def select_strategy(self, task: str, context: str = "",
                       uncertainty: Optional[Uncertainty] = None) -> Strategy:
        """
        Select strategy for task execution.
        
        BACKWARD COMPATIBLE: Same signature and return type as original.
        Would delegate to StrategySelector component.
        """
        # In full implementation, this would be:
        # return self.strategy_selector.select_strategy(task, context, uncertainty)
        
        # For demonstration, return a default strategy
        if uncertainty is None:
            uncertainty = self.assess_uncertainty(task, context)
            
        if uncertainty.level == 'high':
            return self._strategies['decompose_parallel']
        elif uncertainty.level == 'medium':
            return self._strategies['guided_iteration']
        else:
            return self._strategies['direct_execution']
            
    def reflect(self, task: str, result: Any, context: Dict[str, Any]) -> Reflection:
        """
        Reflect on task execution and extract learnings.
        
        BACKWARD COMPATIBLE: Same signature and return type as original.
        Would delegate to ReflectionEngine and trigger SkillExtractor.
        """
        # In full implementation:
        # reflection = self.reflection_engine.reflect(task, result, context)
        # self.skill_extractor.maybe_create_skill(reflection)
        # return reflection
        
        # For demonstration, create a simple reflection
        import datetime
        
        reflection = Reflection(
            task_description=task,
            context=str(context),
            approach=context.get('approach', 'unknown'),
            outcome=str(result),
            lessons_learned=self._extract_simple_lessons(task, result, context),
            timestamp=datetime.datetime.now().isoformat(),
            model_used=context.get('model', 'gpt-4'),
            uncertainty_level=context.get('uncertainty_level', 'medium'),
            task_complexity=context.get('task_complexity', 'medium'),
            was_successful=context.get('success', True)
        )
        
        # Store in database (would be done by ReflectionEngine)
        self._store_reflection(reflection)
        
        return reflection
        
    def get_insights_for_task(self, task: str, limit: int = 5) -> List[str]:
        """
        Get relevant insights from past experiences.
        
        BACKWARD COMPATIBLE: Same signature and return type as original.
        Would delegate to KnowledgeRetriever component.
        """
        # In full implementation:
        # return self.knowledge_retriever.get_insights_for_task(task, limit)
        
        # For demonstration, query database for similar past reflections
        query, params = self.db.build_similarity_query(
            'reflections', 'task_description', task, limit
        )
        
        results = self.db.execute_query(query, params)
        
        insights = []
        for result in results:
            if result['was_successful'] and result['lessons_learned']:
                lessons = self.db.json_deserialize(result['lessons_learned'])
                if lessons:
                    insights.extend(lessons[:1])  # Take first lesson from each
                    
        return insights[:limit]
        
    def get_failure_warnings(self, task: str, approach: str = "") -> List[str]:
        """
        Get warnings based on past failures.
        
        BACKWARD COMPATIBLE: Same signature and return type as original.
        Would delegate to KnowledgeRetriever component.
        """
        # In full implementation:
        # return self.knowledge_retriever.get_failure_warnings(task, approach)
        
        # For demonstration, check failure patterns
        query = """
        SELECT error_pattern, mitigation 
        FROM failure_patterns 
        WHERE task_pattern LIKE ? 
        LIMIT 3
        """
        
        results = self.db.execute_query(query, (f"%{task.split()[0]}%",))
        
        warnings = []
        for result in results:
            warning = f"Past failure: {result['error_pattern']}"
            if result['mitigation']:
                warning += f" - {result['mitigation']}"
            warnings.append(warning)
            
        return warnings
        
    def get_recommended_model(self, task: str, complexity: str = "medium") -> str:
        """
        Get recommended model based on past performance.
        
        BACKWARD COMPATIBLE: Same signature and return type as original.
        Would delegate to KnowledgeRetriever component.
        """
        # In full implementation:
        # return self.knowledge_retriever.get_recommended_model(task, complexity)
        
        # For demonstration, use simple logic
        if complexity == "high":
            return "claude-2"
        elif complexity == "low":
            return "gpt-3.5-turbo"
        else:
            return "gpt-4"
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cognitive system statistics.
        
        BACKWARD COMPATIBLE: Same signature and return type as original.
        Aggregates stats from all components.
        """
        stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM reflections) as total_reflections,
            (SELECT COUNT(*) FROM reflections WHERE was_successful = 1) as successful_tasks,
            (SELECT COUNT(*) FROM strategies) as total_strategies,
            (SELECT COUNT(*) FROM skills) as total_skills,
            (SELECT COUNT(*) FROM failure_patterns) as known_failure_patterns
        """
        
        results = self.db.execute_query(stats_query)
        
        if results:
            stats = results[0]
            # Calculate success rate
            if stats['total_reflections'] > 0:
                stats['success_rate'] = stats['successful_tasks'] / stats['total_reflections']
            else:
                stats['success_rate'] = 0.0
        else:
            stats = {
                'total_reflections': 0,
                'successful_tasks': 0,
                'total_strategies': len(self._strategies),
                'total_skills': 0,
                'known_failure_patterns': 0,
                'success_rate': 0.0
            }
            
        return stats
        
    def shutdown(self) -> None:
        """
        Cleanup resources.
        
        BACKWARD COMPATIBLE: Same signature as original.
        Ensures all components properly close connections.
        """
        self.db.close()
        # Each component would handle its own cleanup:
        # self.strategy_selector.shutdown()
        # self.reflection_engine.shutdown()
        # etc.
        
    # ===== INTERNAL HELPER METHODS =====
    # These demonstrate how internal logic is better organized
    
    def _load_default_strategies(self) -> Dict[str, Strategy]:
        """Load default strategies (would be in StrategySelector)."""
        return {
            'direct_execution': Strategy(
                name='direct_execution',
                description='Execute task directly without decomposition',
                when_to_use=['Simple tasks', 'Clear requirements', 'Low uncertainty'],
                steps=['Understand requirements', 'Execute task', 'Verify result'],
                requires_planning=False,
                supports_delegation=False,
                model_flexibility='any'
            ),
            'decompose_parallel': Strategy(
                name='decompose_parallel',
                description='Break down into parallel subtasks',
                when_to_use=['Complex tasks', 'Multiple independent components'],
                steps=['Analyze task', 'Identify subtasks', 'Execute in parallel', 'Combine results'],
                requires_planning=True,
                supports_delegation=True,
                model_flexibility='advanced'
            ),
            'guided_iteration': Strategy(
                name='guided_iteration',
                description='Iterative approach with guidance',
                when_to_use=['Medium complexity', 'Requires refinement'],
                steps=['Initial attempt', 'Review', 'Refine', 'Finalize'],
                requires_planning=False,
                supports_delegation=False,
                model_flexibility='any'
            )
        }
        
    def _store_reflection(self, reflection: Reflection) -> None:
        """Store reflection in database (would be in ReflectionEngine)."""
        query = """
        INSERT INTO reflections (
            task_description, context, approach, outcome,
            lessons_learned, timestamp, model_used,
            uncertainty_level, task_complexity, was_successful
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            reflection.task_description,
            reflection.context,
            reflection.approach,
            reflection.outcome,
            self.db.json_serialize(reflection.lessons_learned),
            reflection.timestamp,
            reflection.model_used,
            reflection.uncertainty_level,
            reflection.task_complexity,
            1 if reflection.was_successful else 0
        )
        
        self.db.execute_write(query, params)
        
    def _extract_simple_lessons(self, task: str, result: Any, 
                               context: Dict[str, Any]) -> List[str]:
        """Extract simple lessons (would be in ReflectionEngine)."""
        lessons = []
        
        if context.get('success', True):
            lessons.append(f"Successfully completed {task} using {context.get('approach', 'standard approach')}")
            if context.get('model'):
                lessons.append(f"{context['model']} performed well for this type of task")
        else:
            lessons.append(f"Failed to complete {task} - needs different approach")
            if context.get('error'):
                lessons.append(f"Error encountered: {context['error']}")
                
        return lessons


# Global singleton management (BACKWARD COMPATIBLE)
_brain_instance: Optional[CognitiveArchitecture] = None


def get_brain(base_path: str = ".swarm") -> CognitiveArchitecture:
    """
    Get or create the global brain instance.
    
    BACKWARD COMPATIBLE: Exact same function signature and behavior.
    """
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = CognitiveArchitecture(base_path)
    return _brain_instance