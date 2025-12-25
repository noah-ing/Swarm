"""
Reflection Engine Component
===========================

Handles post-task analysis and learning extraction.
Extracted from the monolithic CognitiveArchitecture class.

The ReflectionEngine focuses specifically on:
- Analyzing task outcomes for learning opportunities
- Extracting insights from successful and failed executions
- Coordinating with SkillExtractor for pattern recognition
- Building cognitive confidence scores
"""

import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from .database import CognitiveDatabase, Reflection

if TYPE_CHECKING:
    from .skill_extractor import SkillExtractor


class ReflectionEngine:
    """
    Specialized component for post-task reflection and learning extraction.
    
    Responsibilities:
    - Analyze task outcomes (success/failure patterns)
    - Extract actionable insights from execution data
    - Calculate confidence scores for reflections
    - Coordinate with database for persistence
    - Trigger skill extraction and failure pattern updates
    """

    def __init__(self, db: CognitiveDatabase, skill_extractor: Optional['SkillExtractor'] = None):
        """Initialize with database and optional skill extractor dependency injection."""
        self.db = db
        self.skill_extractor = skill_extractor

    def reflect(
        self,
        task: str,
        outcome: str,
        success: bool,
        model_used: str,
        tokens_used: int,
        duration_seconds: float,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> Reflection:
        """
        Conduct comprehensive post-task reflection to extract learnings.

        This is the main entry point called after every task completion
        to build up cognitive knowledge and improve future performance.

        Args:
            task: The original task description
            outcome: What actually happened
            success: Whether the task succeeded
            model_used: Which AI model was used
            tokens_used: Token consumption count
            duration_seconds: Execution time
            tool_calls: List of tool invocations made
            error: Error message if task failed

        Returns:
            Reflection object containing all extracted insights
        """
        # Initialize analysis containers
        what_worked = []
        what_failed = []
        insights = []
        skill_candidates = []

        if success:
            what_worked, insights, skill_candidates = self._analyze_success(
                task, model_used, tokens_used, duration_seconds, tool_calls
            )
        else:
            what_failed, insights = self._analyze_failure(
                task, model_used, tokens_used, duration_seconds, error
            )

        # Calculate confidence in this reflection
        confidence = self._calculate_confidence(
            success, insights, skill_candidates, tokens_used
        )

        # Create reflection object
        reflection = Reflection(
            task=task,
            outcome=outcome,
            success=success,
            what_worked=what_worked,
            what_failed=what_failed,
            insights=insights,
            skill_candidates=skill_candidates,
            confidence=confidence,
            model_used=model_used,
            tokens_used=tokens_used,
            duration_seconds=duration_seconds,
            created_at=datetime.now(),
        )

        # Store reflection in database
        reflection_id = self.db.store_reflection(reflection)

        # Extract skills if skill_extractor is available and candidates found
        if self.skill_extractor and skill_candidates:
            for candidate in skill_candidates:
                self.skill_extractor.maybe_create_skill(candidate)

        # Update failure patterns if failed and skill_extractor available
        if not success and error and self.skill_extractor:
            self.skill_extractor.update_failure_pattern(task, error)

        return reflection

    def _analyze_success(
        self,
        task: str,
        model_used: str,
        tokens_used: int,
        duration_seconds: float,
        tool_calls: Optional[List[Dict[str, Any]]],
    ) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
        """
        Analyze successful task execution for learning opportunities.

        Returns:
            tuple: (what_worked, insights, skill_candidates)
        """
        what_worked = []
        insights = []
        skill_candidates = []

        # Record successful model usage
        what_worked.append(f"Task completed successfully with {model_used}")

        # Analyze efficiency patterns
        if tokens_used < 5000:
            what_worked.append("Efficient token usage")
            insights.append("Simple approach worked well")

        if duration_seconds < 10:
            what_worked.append("Fast execution")

        # Analyze tool usage patterns for skill extraction
        if tool_calls:
            tool_sequence = [tc.get("name") for tc in tool_calls if tc.get("name")]
            
            # Look for patterns that could become reusable skills
            if len(tool_sequence) <= 5 and len(tool_sequence) >= 2:  # Reasonable skill size
                skill_candidate = {
                    "name": f"skill_{hashlib.md5(task.encode()).hexdigest()[:8]}",
                    "pattern": task[:200],  # Increased pattern length
                    "tools": tool_sequence,
                }
                skill_candidates.append(skill_candidate)
                insights.append(f"Potential skill: {len(tool_sequence)}-step pattern")

        # Analyze task complexity vs. execution simplicity
        task_complexity = self._estimate_task_complexity(task)
        if task_complexity > 0.7 and tokens_used < 10000:
            insights.append("Complex task solved efficiently")
            what_worked.append("Effective problem decomposition")

        return what_worked, insights, skill_candidates

    def _analyze_failure(
        self,
        task: str,
        model_used: str,
        tokens_used: int,
        duration_seconds: float,
        error: Optional[str],
    ) -> tuple[List[str], List[str]]:
        """
        Analyze failed task execution for learning opportunities.

        Returns:
            tuple: (what_failed, insights)
        """
        what_failed = []
        insights = []

        # Record failed model usage
        what_failed.append(f"Task failed with {model_used}")

        # Analyze specific error patterns
        if error:
            error_lower = error.lower()
            
            if "timeout" in error_lower:
                what_failed.append("Execution timed out")
                insights.append("Consider breaking into smaller subtasks")
                
            elif "token" in error_lower or "limit" in error_lower:
                what_failed.append("Token limit exceeded")
                insights.append("Reduce context or use summarization")
                
            elif "not found" in error_lower:
                what_failed.append("Resource not found")
                insights.append("Verify paths/resources before operations")
                
            elif "permission" in error_lower:
                what_failed.append("Permission denied")
                insights.append("Check access rights and file permissions")
                
            elif "syntax" in error_lower:
                what_failed.append("Syntax error in generated code")
                insights.append("Validate code syntax before execution")
                
            else:
                what_failed.append(f"Error: {error[:100]}")
                insights.append("Investigate root cause of error")

        # Analyze resource usage patterns
        if tokens_used > 50000:
            insights.append("High token usage suggests over-complicated approach")
            what_failed.append("Inefficient problem solving strategy")

        if duration_seconds > 120:
            insights.append("Long execution time indicates inefficiency")
            what_failed.append("Slow execution pathway")

        # Provide strategic insights
        task_complexity = self._estimate_task_complexity(task)
        if task_complexity < 0.3:
            insights.append("Simple task failed - check basic prerequisites")
        else:
            insights.append("Complex task failed - consider decomposition strategy")

        return what_failed, insights

    def _calculate_confidence(
        self,
        success: bool,
        insights: List[str],
        skill_candidates: List[Dict[str, Any]],
        tokens_used: int,
    ) -> float:
        """
        Calculate confidence score for the reflection quality.

        Higher confidence indicates more reliable learning extraction.
        """
        # Base confidence depends on success
        confidence = 0.7 if success else 0.5

        # Bonus for extracting insights
        if len(insights) > 0:
            confidence += 0.1

        # Bonus for identifying skill candidates
        if len(skill_candidates) > 0:
            confidence += 0.1

        # Bonus for reasonable token usage (not too low or too high)
        if 1000 < tokens_used < 20000:
            confidence += 0.05

        # Penalty for extremely high token usage (likely confused)
        if tokens_used > 100000:
            confidence -= 0.2

        # Clamp to valid range
        return min(1.0, max(0.0, confidence))

    def _estimate_task_complexity(self, task: str) -> float:
        """
        Estimate task complexity based on linguistic patterns.

        Returns score from 0.0 (simple) to 1.0 (complex).
        """
        task_lower = task.lower()
        complexity_score = 0.0

        # Word count indicates complexity
        word_count = len(task.split())
        complexity_score += min(0.3, word_count / 100)

        # Complex operations
        complex_words = [
            "refactor", "architect", "design", "implement", "integrate",
            "optimize", "analyze", "transform", "migrate", "deploy"
        ]
        complexity_score += 0.1 * sum(1 for word in complex_words if word in task_lower)

        # Multiple file operations
        if task_lower.count('.') > 2:
            complexity_score += 0.2

        # Logical complexity indicators
        logical_words = ["if", "when", "unless", "depending", "based on", "according to"]
        complexity_score += 0.05 * sum(1 for word in logical_words if word in task_lower)

        # Technical depth indicators
        tech_words = ["database", "api", "algorithm", "pattern", "framework", "library"]
        complexity_score += 0.1 * sum(1 for word in tech_words if word in task_lower)

        return min(1.0, complexity_score)

    def get_insights_for_task(self, task: str) -> List[str]:
        """
        Get relevant insights from past reflections for a similar task.

        Delegates to database but provides the interface for cognitive components.
        """
        return self.db.get_insights_for_task(task)

    def get_failure_warnings(self, task: str) -> List[str]:
        """
        Get warnings based on past failure patterns for a task.

        Delegates to database but provides the interface for cognitive components.
        """
        return self.db.get_failure_warnings_for_task(task)

    def get_recommended_model(self, task: str, default: str = "sonnet") -> str:
        """
        Recommend a model based on past performance with similar tasks.

        Delegates to database but provides the interface for cognitive components.
        """
        return self.db.get_recommended_model(default)

    def get_reflection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the reflection system's performance.
        
        Returns metrics useful for monitoring reflection quality and system health.
        """
        stats = self.db.get_system_stats()
        
        # Add reflection-specific metrics
        if stats["total_reflections"] > 0:
            stats["success_rate"] = (
                stats["successful_reflections"] / stats["total_reflections"]
            )
        else:
            stats["success_rate"] = 0.0

        return stats