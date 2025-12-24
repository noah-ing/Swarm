"""
Thinker Agent: Meta-cognitive reasoning before action.

The Thinker doesn't execute tasks - it thinks about HOW to approach them.
It considers:
- What strategy to use
- What could go wrong
- What information is missing
- What the success criteria are
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .base import BaseAgent
from brain import get_brain, Uncertainty, Strategy


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{name}.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


THINKER_SYSTEM_PROMPT = """You are the Thinker - a meta-cognitive agent that reasons about HOW to approach tasks.

You do NOT execute tasks. You THINK about them.

## Your Role

Before any task is executed, you analyze:
1. What is actually being asked?
2. What's the best approach?
3. What could go wrong?
4. What information is missing?
5. How will we know if it succeeded?

## Output Format

Respond with a JSON analysis:

```json
{
    "understanding": "What the task is really asking for",
    "approach": {
        "strategy": "direct|decompose_parallel|decompose_sequential|hierarchical|explore_first|trial_and_error",
        "reasoning": "Why this strategy",
        "steps": ["Step 1", "Step 2", ...]
    },
    "risks": [
        {"risk": "What could go wrong", "mitigation": "How to prevent/handle it"}
    ],
    "missing_info": ["Information that would help but isn't provided"],
    "assumptions": ["Assumptions we're making"],
    "success_criteria": ["How to verify the task is complete"],
    "confidence": 0.0-1.0,
    "recommended_model": "haiku|sonnet|opus",
    "estimated_complexity": "low|medium|high"
}
```

## Thinking Guidelines

- Be skeptical: What could the user REALLY mean?
- Be cautious: What's the worst case scenario?
- Be efficient: What's the simplest approach that works?
- Be specific: Vague plans lead to vague results

## Strategy Descriptions

- **direct**: Single agent, straightforward execution
- **decompose_parallel**: Split into independent subtasks, run simultaneously
- **decompose_sequential**: Split into dependent subtasks, run in order
- **hierarchical**: Multi-level goal decomposition for very complex tasks
- **explore_first**: Gather information before acting
- **trial_and_error**: Try approaches, learn from results

## Examples

Task: "Fix the bug in login.py"
- Understanding: User wants a specific bug fixed in a specific file
- Strategy: direct (single file, single issue)
- Risks: Bug might be a symptom of deeper issue
- Missing: What bug? Error message? Expected behavior?
- Confidence: 0.6 (need more info but can explore)

Task: "Refactor the authentication system"
- Understanding: Major structural changes to auth code
- Strategy: decompose_sequential (must understand before changing)
- Risks: Breaking existing functionality, security implications
- Missing: Current pain points, target architecture, constraints
- Confidence: 0.4 (high complexity, need exploration)
"""


@dataclass
class ThinkingResult:
    """Result of thinking about a task."""
    understanding: str
    strategy: str
    strategy_reasoning: str
    steps: list[str]
    risks: list[dict]
    missing_info: list[str]
    assumptions: list[str]
    success_criteria: list[str]
    confidence: float
    recommended_model: str
    estimated_complexity: str
    raw_analysis: dict = field(default_factory=dict)


class ThinkerAgent(BaseAgent):
    """
    Meta-cognitive agent that thinks before acting.

    The Thinker analyzes tasks to determine the best approach,
    identify risks, and set success criteria.
    """

    def __init__(self, model: str = "sonnet"):
        super().__init__(model=model, system_prompt=THINKER_SYSTEM_PROMPT)
        self.brain = get_brain()

    def think(self, task: str, context: str = "") -> ThinkingResult:
        """
        Think deeply about how to approach a task.

        Args:
            task: The task to think about
            context: Optional context (codebase info, history, etc.)

        Returns:
            ThinkingResult with analysis and recommendations
        """
        # Get brain's initial assessment
        uncertainty = self.brain.assess_uncertainty(task, context)
        suggested_strategy = self.brain.select_strategy(task, context, uncertainty)
        insights = self.brain.get_insights_for_task(task)
        warnings = self.brain.get_failure_warnings(task)

        # Build prompt with brain's knowledge
        content = f"""## Task to Analyze

{task}

## Brain's Initial Assessment

**Uncertainty Level:** {uncertainty.level} ({uncertainty.score:.0%})
**Uncertainty Reasons:** {', '.join(uncertainty.reasons)}
**Suggested Strategy:** {suggested_strategy.name}
**Recommended Action:** {uncertainty.recommended_action}

"""

        if context:
            content += f"""## Context

{context}

"""

        if insights:
            content += f"""## Insights from Past Experience

{chr(10).join('- ' + i for i in insights)}

"""

        if warnings:
            content += f"""## Warnings from Past Failures

{chr(10).join('- ' + w for w in warnings)}

"""

        content += """
Now, think deeply about this task. Consider all angles.
What's the best way to approach it? What could go wrong?
Respond with your analysis as JSON.
"""

        messages = [{"role": "user", "content": content}]
        response = self.chat(messages=messages)

        # Parse response
        try:
            text = response.content or "{}"

            # Extract JSON
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
            else:
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    text = json_match.group(0)

            analysis = json.loads(text)

            return ThinkingResult(
                understanding=analysis.get("understanding", task),
                strategy=analysis.get("approach", {}).get("strategy", suggested_strategy.name),
                strategy_reasoning=analysis.get("approach", {}).get("reasoning", ""),
                steps=analysis.get("approach", {}).get("steps", suggested_strategy.steps),
                risks=analysis.get("risks", []),
                missing_info=analysis.get("missing_info", []),
                assumptions=analysis.get("assumptions", []),
                success_criteria=analysis.get("success_criteria", ["Task completes without error"]),
                confidence=analysis.get("confidence", 0.5),
                recommended_model=analysis.get("recommended_model", "sonnet"),
                estimated_complexity=analysis.get("estimated_complexity", "medium"),
                raw_analysis=analysis,
            )

        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to brain's suggestion
            return ThinkingResult(
                understanding=task,
                strategy=suggested_strategy.name.lower().replace(" ", "_"),
                strategy_reasoning=f"Brain suggested: {suggested_strategy.description}",
                steps=suggested_strategy.steps,
                risks=[{"risk": "Analysis failed", "mitigation": "Proceeding with default approach"}],
                missing_info=[],
                assumptions=["Using brain's default strategy"],
                success_criteria=["Task completes without error"],
                confidence=0.5,
                recommended_model="sonnet",
                estimated_complexity="medium",
                raw_analysis={},
            )

    def should_ask_user(self, thinking: ThinkingResult) -> tuple[bool, str]:
        """
        Determine if we should ask the user for clarification.

        Returns:
            (should_ask, reason)
        """
        # Very low confidence
        if thinking.confidence < 0.3:
            return True, f"Low confidence ({thinking.confidence:.0%}) in understanding the task"

        # Critical missing information
        critical_missing = [m for m in thinking.missing_info if any(
            word in m.lower() for word in ["which", "what", "where", "password", "credentials", "api key"]
        )]
        if critical_missing:
            return True, f"Missing critical information: {critical_missing[0]}"

        # High risk without mitigation
        unmitigated_risks = [r for r in thinking.risks if not r.get("mitigation")]
        if unmitigated_risks:
            return True, f"High-risk operation without clear mitigation: {unmitigated_risks[0].get('risk', 'Unknown')}"

        return False, ""

    def run(self, task: str, context: str = "") -> ThinkingResult:
        """Run the thinker (alias for think())."""
        return self.think(task, context)
