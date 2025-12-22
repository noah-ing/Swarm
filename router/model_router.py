"""Model router for intelligent model selection."""

import re
from typing import Literal

from config import get_settings


class ModelRouter:
    """Selects the appropriate model based on task complexity."""

    # Keywords suggesting higher complexity
    HIGH_COMPLEXITY_KEYWORDS = {
        "refactor", "architect", "design", "implement", "create",
        "build", "develop", "analyze", "optimize", "debug",
        "security", "performance", "complex", "system",
    }

    # Keywords suggesting lower complexity
    LOW_COMPLEXITY_KEYWORDS = {
        "fix", "typo", "rename", "comment", "format", "lint",
        "simple", "update", "change", "add", "remove", "delete",
        "read", "list", "show", "print", "log",
    }

    # Model tiers (cheapest to most expensive)
    ANTHROPIC_TIERS = ["haiku", "sonnet", "opus"]
    OPENAI_TIERS = ["gpt-4o-mini", "gpt-4o", "o1"]

    def __init__(self, prefer_provider: str | None = None):
        self.settings = get_settings()
        self.prefer_provider = prefer_provider or self.settings.prefer_provider

    def estimate_complexity(self, task: str) -> float:
        """
        Estimate task complexity from 0.0 to 1.0.

        Args:
            task: Task description

        Returns:
            Complexity score (0.0 = trivial, 1.0 = very complex)
        """
        task_lower = task.lower()
        words = set(re.findall(r'\w+', task_lower))

        # Count keyword matches
        high_matches = len(words & self.HIGH_COMPLEXITY_KEYWORDS)
        low_matches = len(words & self.LOW_COMPLEXITY_KEYWORDS)

        # Base complexity from keyword balance
        if high_matches + low_matches == 0:
            keyword_score = 0.5
        else:
            keyword_score = high_matches / (high_matches + low_matches)

        # Adjust for task length (longer = usually more complex)
        length_score = min(1.0, len(task) / 500)

        # Adjust for file references (more files = more complex)
        file_refs = len(re.findall(r'[\w/]+\.\w+', task))
        file_score = min(1.0, file_refs / 5)

        # Weighted combination
        complexity = (
            keyword_score * 0.5 +
            length_score * 0.3 +
            file_score * 0.2
        )

        return complexity

    def select(
        self,
        task: str,
        complexity: str | None = None,
        force_provider: str | None = None,
    ) -> str:
        """
        Select the best model for a task.

        Args:
            task: Task description
            complexity: Override complexity ("low", "medium", "high")
            force_provider: Force a specific provider

        Returns:
            Model name (e.g., "haiku", "gpt-4o")
        """
        # Determine complexity
        if complexity == "low":
            score = 0.2
        elif complexity == "high":
            score = 0.8
        elif complexity == "medium":
            score = 0.5
        else:
            score = self.estimate_complexity(task)

        # Select tier
        provider = force_provider or self.prefer_provider
        tiers = self.ANTHROPIC_TIERS if provider == "anthropic" else self.OPENAI_TIERS

        if score < 0.3:
            return tiers[0]  # Cheapest
        elif score < 0.7:
            return tiers[1]  # Middle
        else:
            return tiers[2]  # Best

    def escalate(self, current_model: str) -> str:
        """
        Escalate to a more powerful model after failure.

        Args:
            current_model: Current model that failed

        Returns:
            Next tier model, or same if already at max
        """
        # Find current tier
        if current_model in self.ANTHROPIC_TIERS:
            tiers = self.ANTHROPIC_TIERS
        elif current_model in self.OPENAI_TIERS:
            tiers = self.OPENAI_TIERS
        else:
            # Unknown model, return as-is
            return current_model

        try:
            idx = tiers.index(current_model)
            if idx < len(tiers) - 1:
                return tiers[idx + 1]
        except ValueError:
            pass

        return current_model

    def get_cost_estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a model and token counts.

        Args:
            model: Model name
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens

        Returns:
            Estimated cost in USD
        """
        costs = self.settings.cost_per_million.get(model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        return input_cost + output_cost
