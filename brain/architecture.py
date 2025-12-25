"""
Refactored Cognitive Architecture
=================================

Main orchestrator that integrates all cognitive components while maintaining
the exact same public interface and singleton pattern as the original brain.py.
"""

from typing import Any, Dict, List, Optional

from brain.database import CognitiveDatabase, Reflection
from brain.uncertainty import UncertaintyAssessor, Uncertainty
from brain.strategy import StrategySelector, Strategy
from brain.reflection import ReflectionEngine
from brain.skill_extractor import SkillExtractor
from brain.knowledge import KnowledgeRetriever


class CognitiveArchitecture:
    """
    The brain of Swarm - handles meta-cognition and learning.

    This is the refactored version that orchestrates all cognitive components
    while maintaining 100% backward compatibility with the original brain.py.

    Key improvements:
    - Separated concerns into focused components
    - Better testability and maintainability
    - Same exact public API as original
    """

    def __init__(self, db_path: str | None = None):
        """Initialize with backward-compatible database path."""
        # Initialize centralized database
        self.db = CognitiveDatabase(db_path)
        self.db_path = self.db.db_path  # Maintain compatibility

        # Initialize all cognitive components with dependency injection
        self.uncertainty_assessor = UncertaintyAssessor(self.db)
        self.strategy_selector = StrategySelector(self.db)
        self.reflection_engine = ReflectionEngine(self.db)
        self.skill_extractor = SkillExtractor(self.db)
        self.knowledge_retriever = KnowledgeRetriever(self.db)

        # Maintain backward compatibility
        self.strategies = self.strategy_selector.strategies

        # Import settings for compatibility
        try:
            from config import get_settings
            self.settings = get_settings()
        except ImportError:
            # Handle case where config might not be available
            self.settings = None

    def assess_uncertainty(self, task: str, context: str = "") -> Uncertainty:
        """
        Assess uncertainty level for a task.

        Delegates to UncertaintyAssessor component.
        Maintains exact same interface as original.
        """
        return self.uncertainty_assessor.assess_uncertainty(task, context)

    def select_strategy(self, task: str, context: str = "", uncertainty: Uncertainty | None = None) -> Strategy:
        """
        Select the best strategy for a task using meta-cognition.

        Delegates to StrategySelector component.
        Maintains exact same interface as original.
        """
        return self.strategy_selector.select_strategy(task, context, uncertainty)

    def reflect(
        self,
        task: str,
        outcome: str,
        success: bool,
        model_used: str,
        tokens_used: int,
        duration_seconds: float,
        tool_calls: list[dict] | None = None,
        error: str | None = None,
    ) -> Reflection:
        """
        Reflect on a completed task to extract learnings.

        Delegates to ReflectionEngine component.
        Maintains exact same interface as original.
        """
        return self.reflection_engine.reflect(
            task=task,
            outcome=outcome,
            success=success,
            model_used=model_used,
            tokens_used=tokens_used,
            duration_seconds=duration_seconds,
            tool_calls=tool_calls,
            error=error
        )

    def get_insights_for_task(self, task: str) -> list[str]:
        """
        Get relevant insights from past reflections for a task.

        Delegates to KnowledgeRetriever component.
        Maintains exact same interface as original.
        """
        return self.knowledge_retriever.get_insights_for_task(task)

    def get_failure_warnings(self, task: str) -> list[str]:
        """
        Get warnings based on past failure patterns.

        Delegates to KnowledgeRetriever component.
        Maintains exact same interface as original.
        """
        return self.knowledge_retriever.get_failure_warnings(task)

    def get_recommended_model(self, task: str, default: str = "sonnet") -> str:
        """
        Recommend a model based on past performance with similar tasks.

        Delegates to KnowledgeRetriever component.
        Maintains exact same interface as original.
        """
        return self.knowledge_retriever.get_recommended_model(task, default)

    def get_stats(self) -> dict[str, Any]:
        """
        Get cognitive system statistics.

        Delegates to CognitiveDatabase component.
        Maintains exact same interface as original.
        """
        return self.db.get_system_stats()

    def calculate_risk_score(self, metrics: List[Dict[str, Any]]) -> float:
        """
        Calculate a weighted risk score based on multiple metrics.

        Args:
            metrics (List[Dict[str, Any]]): List of dictionaries containing metric details.
                Each metric should have:
                - 'name': str, name of the metric
                - 'value': float/int, numeric value of the metric
                - 'weight': float, importance of this metric (default: 1.0)
                - 'threshold': float (optional), risk threshold value

        Returns:
            float: A normalized risk score between 0.0 and 1.0
        """
        if not metrics:
            return 0.0

        total_risk = 0.0
        total_weight = 0.0

        for metric in metrics:
            # Default weight to 1.0 if not provided
            weight = metric.get('weight', 1.0)
            value = metric.get('value', 0.0)

            # Optional threshold logic
            if 'threshold' in metric:
                # Normalized distance from threshold
                normalized_risk = max(0.0, value - metric['threshold']) / max(metric['threshold'], 1.0)
                metric_risk = normalized_risk * weight
            else:
                # If no threshold, normalize raw value
                metric_risk = min(1.0, abs(value) / max(abs(value) + 1, 1.0)) * weight

            total_risk += metric_risk
            total_weight += weight

        # Normalize by total weight
        risk_score = total_risk / max(total_weight, 1.0)
        return min(1.0, max(0.0, risk_score))  # Clamp between 0 and 1

    # Private methods for backward compatibility
    def _init_db(self):
        """Initialize the cognitive database - provided for compatibility."""
        # Already initialized in __init__, this is a no-op for compatibility
        pass

    def _load_strategies(self):
        """Load built-in strategies - provided for compatibility."""
        # Already loaded by StrategySelector, this is a no-op for compatibility
        pass

    def _has_similar_past_solution(self, task: str) -> bool:
        """Check if we have a similar past solution - provided for compatibility."""
        return self.db.has_similar_past_solution(task)

    def _store_reflection(self, reflection: Reflection):
        """Store a reflection in the database - provided for compatibility."""
        self.db.store_reflection(reflection)

    def _maybe_create_skill(self, candidate: dict):
        """Create a skill if the pattern is seen multiple times - provided for compatibility."""
        self.skill_extractor.maybe_create_skill(candidate)

    def _update_failure_pattern(self, task: str, error: str):
        """Track failure patterns to avoid repeating mistakes - provided for compatibility."""
        self.skill_extractor.update_failure_pattern(task, error)

    # Additional methods that enhance the architecture while maintaining compatibility

    def get_component_health(self) -> Dict[str, Any]:
        """
        Get health status of all cognitive components.

        This is a new method that doesn't break backward compatibility.
        """
        stats = self.get_stats()
        return {
            'database': {
                'status': 'healthy',
                'total_records': sum([
                    stats.get('total_reflections', 0),
                    stats.get('skills_learned', 0),
                    stats.get('failure_patterns_tracked', 0)
                ])
            },
            'components': {
                'uncertainty_assessor': 'active',
                'strategy_selector': 'active',
                'reflection_engine': 'active',
                'skill_extractor': 'active',
                'knowledge_retriever': 'active'
            },
            'performance': {
                'avg_confidence': stats.get('avg_confidence', 0),
                'success_rate': (
                    stats.get('successful_reflections', 0) / max(stats.get('total_reflections', 1), 1)
                )
            }
        }

    def get_contextual_guidance(self, task: str) -> Dict[str, Any]:
        """
        Get comprehensive guidance for a task combining all components.

        This is a new method that showcases the power of the refactored architecture.
        """
        # Assess uncertainty
        uncertainty = self.assess_uncertainty(task)

        # Get strategy recommendations
        strategy_rec = self.strategy_selector.get_strategy_recommendations(task)

        # Get contextual knowledge
        knowledge = self.knowledge_retriever.get_contextual_knowledge(task)

        # Get skill suggestions
        relevant_skills = self.knowledge_retriever.get_skill_suggestions(task)

        return {
            'uncertainty': {
                'level': uncertainty.level,
                'score': uncertainty.score,
                'reasons': uncertainty.reasons,
                'recommended_action': uncertainty.recommended_action
            },
            'strategy': {
                'primary': strategy_rec['primary_strategy'].name,
                'alternatives': [s.name for s in strategy_rec['alternative_strategies']],
                'rationale': strategy_rec['selection_rationale']
            },
            'knowledge': knowledge,
            'relevant_skills': relevant_skills[:3] if relevant_skills else [],
            'recommended_model': knowledge['recommended_model']
        }

    def __del__(self):
        """Cleanup database connection on deletion."""
        if hasattr(self, 'db'):
            self.db.close()


# Global brain instance
_brain: CognitiveArchitecture | None = None


def get_brain() -> CognitiveArchitecture:
    """Get or create the global brain instance."""
    global _brain
    if _brain is None:
        _brain = CognitiveArchitecture()
    return _brain