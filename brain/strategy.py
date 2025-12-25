"""
Strategy Selector Component
===========================

Manages strategy selection and loading.
Extracted from CognitiveArchitecture for better modularity.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional

from brain.database import CognitiveDatabase
from brain.uncertainty import Uncertainty, UncertaintyAssessor


@dataclass
class Strategy:
    """A strategy for approaching a task."""
    name: str
    description: str
    applicable_patterns: list[str]
    steps: list[str]
    success_rate: float = 0.0
    uses: int = 0


class StrategySelector:
    """
    Manages strategy selection and loading.
    
    Extracted from CognitiveArchitecture to provide focused strategy management
    with built-in strategies and historical performance tracking.
    """
    
    def __init__(self, db: CognitiveDatabase):
        """Initialize with database and load built-in strategies."""
        self.db = db
        self.strategies: Dict[str, Strategy] = {}
        self._load_strategies()
    
    def _load_strategies(self) -> None:
        """Load built-in strategies - extracted from CognitiveArchitecture."""
        self.strategies = {
            "direct_execution": Strategy(
                name="Direct Execution",
                description="Run task directly with single agent",
                applicable_patterns=["simple", "query", "single file", "command"],
                steps=["Analyze task", "Select tool", "Execute", "Verify"],
                success_rate=0.85,
            ),
            "decompose_parallel": Strategy(
                name="Parallel Decomposition",
                description="Split into independent subtasks, run in parallel",
                applicable_patterns=["multiple files", "batch", "independent operations"],
                steps=["Identify components", "Create subtasks", "Run parallel", "Aggregate"],
                success_rate=0.75,
            ),
            "decompose_sequential": Strategy(
                name="Sequential Decomposition",
                description="Split into dependent subtasks, run in order",
                applicable_patterns=["pipeline", "dependent", "build on previous"],
                steps=["Identify phases", "Order by dependency", "Execute sequentially", "Chain results"],
                success_rate=0.70,
            ),
            "explore_first": Strategy(
                name="Explore First",
                description="Understand before acting",
                applicable_patterns=["unfamiliar", "complex", "codebase", "debug"],
                steps=["Search for patterns", "Read relevant files", "Build mental model", "Plan action", "Execute"],
                success_rate=0.80,
            ),
            "trial_and_error": Strategy(
                name="Trial and Error",
                description="Try approaches, learn from failures",
                applicable_patterns=["unclear requirements", "experimental", "multiple solutions"],
                steps=["Generate hypothesis", "Test quickly", "Analyze result", "Iterate or succeed"],
                success_rate=0.65,
            ),
            "template_match": Strategy(
                name="Template Match",
                description="Find similar past solution, adapt it",
                applicable_patterns=["common task", "seen before", "standard pattern"],
                steps=["Search memory", "Find similar", "Adapt template", "Execute"],
                success_rate=0.90,
            ),
        }
    
    def select_strategy(self, task: str, context: str = "", uncertainty: Optional[Uncertainty] = None) -> Strategy:
        """
        Select the best strategy for a task using meta-cognition.
        
        This is the core method extracted from CognitiveArchitecture that implements
        the strategy selection logic based on task patterns and uncertainty analysis.
        
        Args:
            task: The task description
            context: Additional context information
            uncertainty: Pre-computed uncertainty assessment (optional)
            
        Returns:
            The selected Strategy object
        """
        task_lower = task.lower()
        
        # Check uncertainty first
        if uncertainty is None:
            assessor = UncertaintyAssessor(self.db)
            uncertainty = assessor.assess_uncertainty(task, context)
        
        # If high uncertainty, prefer exploratory strategies
        if uncertainty.level in ("high", "critical"):
            return self.strategies["explore_first"]
        
        # Check for template match opportunity (fastest path)
        if self._has_similar_past_solution(task):
            return self.strategies["template_match"]
        
        # Analyze task characteristics - reordered for better priority
        words = task_lower.split()
        
        # Pipeline/sequential operations (check this early before other patterns)
        sequential_indicators = ["then", "after", "first", "next", "finally", "step"]
        if any(s in task_lower for s in sequential_indicators):
            return self.strategies["decompose_sequential"]
        
        # Multiple independent operations
        if " and " in task_lower or ", " in task_lower:
            # Check if they're independent and not sequential
            parts = re.split(r' and |, ', task_lower)
            if len(parts) >= 2 and all(len(p.split()) < 10 for p in parts):
                # Make sure it's not actually sequential
                has_sequential = any(s in task_lower for s in sequential_indicators)
                if not has_sequential:
                    return self.strategies["decompose_parallel"]
        
        # Simple queries/commands
        simple_indicators = ["list", "show", "print", "count", "what", "where", "find", "read"]
        if any(task_lower.startswith(s) for s in simple_indicators):
            return self.strategies["direct_execution"]
        
        # Single file operations (be more specific to avoid false matches)
        if (re.search(r'fix|edit|change|modify', task_lower) and 
            task_lower.count('.') <= 2 and 
            not any(s in task_lower for s in sequential_indicators) and
            len(words) < 15):  # Simple single file operations
            return self.strategies["direct_execution"]
        
        # Complex/unfamiliar tasks
        complex_indicators = ["refactor", "architect", "design", "implement", "build", "create"]
        if any(c in task_lower for c in complex_indicators):
            if len(words) > 15:
                return self.strategies["explore_first"]
            return self.strategies["decompose_sequential"]
        
        # Default to direct for simple, explore for complex
        if len(words) < 10:
            return self.strategies["direct_execution"]
        return self.strategies["explore_first"]
    
    def _has_similar_past_solution(self, task: str) -> bool:
        """Check if we have a similar past solution - extracted from CognitiveArchitecture."""
        return self.db.has_similar_past_solution(task)
    
    def get_all_strategies(self) -> Dict[str, Strategy]:
        """Get all available strategies."""
        return self.strategies.copy()
    
    def get_strategy_by_name(self, name: str) -> Optional[Strategy]:
        """Get a specific strategy by name."""
        return self.strategies.get(name)
    
    def get_strategies_for_pattern(self, pattern: str) -> list[Strategy]:
        """
        Get strategies that are applicable for a specific pattern.
        
        Args:
            pattern: Pattern to match against strategy applicable_patterns
            
        Returns:
            List of applicable strategies sorted by success rate
        """
        pattern_lower = pattern.lower()
        applicable = []
        
        for strategy in self.strategies.values():
            if any(p in pattern_lower for p in strategy.applicable_patterns):
                applicable.append(strategy)
        
        # Sort by success rate, descending
        return sorted(applicable, key=lambda s: s.success_rate, reverse=True)
    
    def get_strategy_recommendations(self, task: str, context: str = "") -> dict:
        """
        Get detailed strategy recommendations with rationale.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Dict with recommended strategy and alternatives with explanations
        """
        # Assess uncertainty first
        assessor = UncertaintyAssessor(self.db)
        uncertainty = assessor.assess_uncertainty(task, context)
        
        # Select primary strategy
        primary_strategy = self.select_strategy(task, context, uncertainty)
        
        # Get alternative strategies based on task patterns
        task_lower = task.lower()
        alternatives = []
        
        # Add uncertainty-based alternatives
        for fallback in uncertainty.fallback_strategies:
            if fallback in self.strategies and fallback != primary_strategy.name.lower().replace(" ", "_"):
                alternatives.append(self.strategies[fallback])
        
        # Add pattern-based alternatives
        pattern_strategies = self.get_strategies_for_pattern(task)
        for strategy in pattern_strategies[:2]:  # Top 2 pattern matches
            if strategy.name != primary_strategy.name and strategy not in alternatives:
                alternatives.append(strategy)
        
        return {
            'primary_strategy': primary_strategy,
            'uncertainty_assessment': uncertainty,
            'alternative_strategies': alternatives[:3],  # Limit to 3 alternatives
            'selection_rationale': self._get_selection_rationale(task, primary_strategy, uncertainty)
        }
    
    def _get_selection_rationale(self, task: str, strategy: Strategy, uncertainty: Uncertainty) -> str:
        """Generate human-readable rationale for strategy selection."""
        task_lower = task.lower()
        reasons = []
        
        # Uncertainty-based reasoning
        if uncertainty.level in ("high", "critical"):
            reasons.append(f"High uncertainty ({uncertainty.level}) suggests exploratory approach")
        elif uncertainty.level == "low":
            reasons.append("Low uncertainty allows for direct execution")
        
        # Pattern-based reasoning
        if self._has_similar_past_solution(task):
            reasons.append("Similar past solutions available for template matching")
        
        # Sequential pattern detection
        sequential_indicators = ["then", "after", "first", "next", "finally", "step"]
        if any(s in task_lower for s in sequential_indicators):
            reasons.append("Sequential steps detected")
        
        # Task characteristics
        if len(task_lower.split()) < 10:
            reasons.append("Simple task structure suggests direct approach")
        elif len(task_lower.split()) > 20:
            reasons.append("Complex task requires structured approach")
        
        # Specific patterns
        if " and " in task_lower or ", " in task_lower:
            reasons.append("Multiple operations detected")
        
        if not reasons:
            reasons.append(f"Default selection based on task complexity")
        
        return f"Selected {strategy.name}: " + "; ".join(reasons)
    
    def update_strategy_performance(self, strategy_name: str, success: bool) -> None:
        """
        Update strategy performance statistics.
        
        Args:
            strategy_name: Name of the strategy that was used
            success: Whether the strategy was successful
        """
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            strategy.uses += 1
            
            # Update success rate using running average
            if strategy.uses == 1:
                strategy.success_rate = 1.0 if success else 0.0
            else:
                # Weighted update to give more weight to recent performance
                weight = 0.1  # 10% weight to new result
                if success:
                    strategy.success_rate = (1 - weight) * strategy.success_rate + weight * 1.0
                else:
                    strategy.success_rate = (1 - weight) * strategy.success_rate + weight * 0.0
                
                # Ensure success rate stays in bounds
                strategy.success_rate = max(0.0, min(1.0, strategy.success_rate))