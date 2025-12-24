"""
CriticAgent: Reviews and critiques solutions from other agents.

Provides structured feedback on:
- Correctness issues
- Efficiency concerns
- Maintainability problems
- Security risks
"""

import json
from dataclasses import dataclass, field
from typing import Any

from .base import BaseAgent, Message


@dataclass
class Issue:
    """An identified issue in a solution."""
    description: str
    severity: str  # low, medium, high, critical
    category: str  # correctness, efficiency, maintainability, security
    line_hint: str = ""  # Optional hint about where the issue is
    suggestion: str = ""  # How to fix it


@dataclass
class CritiqueResult:
    """Result of critiquing a solution."""
    issues: list[Issue]
    score: float  # 0.0-1.0 overall quality score
    dimensions: dict[str, float]  # Per-dimension scores
    summary: str  # Brief summary of the critique
    strengths: list[str]  # What's good about the solution
    suggestions: list[str]  # Improvement ideas
    severity: str  # Overall severity: low, medium, high, critical
    input_tokens: int = 0
    output_tokens: int = 0


CRITIC_SYSTEM_PROMPT = """You are a senior code reviewer and solution critic. Your role is to provide thorough, constructive feedback on proposed solutions.

When reviewing a solution, analyze it across these dimensions:
1. **Correctness**: Does it solve the problem correctly? Are there edge cases missed?
2. **Efficiency**: Is it reasonably efficient? Are there obvious performance issues?
3. **Maintainability**: Is the code clean and maintainable? Are there design issues?
4. **Security**: Are there any security concerns? Input validation issues?

Be specific and actionable. Don't just point out problems - suggest solutions.

Respond in this exact JSON format:
{
    "issues": [
        {
            "description": "Clear description of the issue",
            "severity": "low|medium|high|critical",
            "category": "correctness|efficiency|maintainability|security",
            "line_hint": "Optional: where in the code",
            "suggestion": "How to fix it"
        }
    ],
    "dimensions": {
        "correctness": 0.0-1.0,
        "efficiency": 0.0-1.0,
        "maintainability": 0.0-1.0,
        "security": 0.0-1.0
    },
    "strengths": ["Good aspects of the solution"],
    "suggestions": ["General improvement ideas"],
    "summary": "One paragraph summary of your critique"
}"""


class CriticAgent(BaseAgent):
    """
    Reviews and critiques solutions from other agents.

    Provides structured feedback to help improve solution quality
    through multi-agent negotiation.
    """

    def __init__(self, model: str = "sonnet"):
        super().__init__(model=model)
        self.system_prompt = CRITIC_SYSTEM_PROMPT

    def critique(
        self,
        task: str,
        proposed_solution: str,
        context: str = "",
    ) -> CritiqueResult:
        """
        Critique a proposed solution.

        Args:
            task: The original task description
            proposed_solution: The solution to critique
            context: Additional context about the codebase/requirements

        Returns:
            CritiqueResult with structured feedback
        """
        user_content = f"""## Task
{task}

## Proposed Solution
{proposed_solution}"""

        if context:
            user_content += f"\n\n## Context\n{context}"

        user_content += "\n\nProvide your critique in the specified JSON format."

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_content),
        ]

        response = self._call_llm(messages)

        return self._parse_critique(response.content, response.input_tokens, response.output_tokens)

    def _parse_critique(
        self,
        content: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CritiqueResult:
        """Parse the LLM response into a CritiqueResult."""
        try:
            # Extract JSON from response
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            # Parse issues
            issues = []
            for issue_data in data.get("issues", []):
                issues.append(Issue(
                    description=issue_data.get("description", ""),
                    severity=issue_data.get("severity", "medium"),
                    category=issue_data.get("category", "correctness"),
                    line_hint=issue_data.get("line_hint", ""),
                    suggestion=issue_data.get("suggestion", ""),
                ))

            # Get dimensions
            dimensions = data.get("dimensions", {
                "correctness": 0.5,
                "efficiency": 0.5,
                "maintainability": 0.5,
                "security": 0.5,
            })

            # Calculate overall score
            score = sum(dimensions.values()) / len(dimensions) if dimensions else 0.5

            # Determine overall severity from issues
            severities = [i.severity for i in issues]
            if "critical" in severities:
                overall_severity = "critical"
            elif "high" in severities:
                overall_severity = "high"
            elif "medium" in severities:
                overall_severity = "medium"
            else:
                overall_severity = "low"

            return CritiqueResult(
                issues=issues,
                score=score,
                dimensions=dimensions,
                summary=data.get("summary", "No summary provided"),
                strengths=data.get("strengths", []),
                suggestions=data.get("suggestions", []),
                severity=overall_severity,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return a default critique if parsing fails
            return CritiqueResult(
                issues=[Issue(
                    description="Unable to parse critique response",
                    severity="low",
                    category="correctness",
                )],
                score=0.5,
                dimensions={
                    "correctness": 0.5,
                    "efficiency": 0.5,
                    "maintainability": 0.5,
                    "security": 0.5,
                },
                summary=content[:500] if content else "Critique parsing failed",
                strengths=[],
                suggestions=[],
                severity="low",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    def format_critique(self, result: CritiqueResult) -> str:
        """Format a critique result for display or context injection."""
        lines = [f"## Critique (Score: {result.score:.0%})"]
        lines.append(f"\n{result.summary}")

        if result.strengths:
            lines.append("\n### Strengths")
            for s in result.strengths:
                lines.append(f"- {s}")

        if result.issues:
            lines.append("\n### Issues")
            for issue in result.issues:
                lines.append(f"- [{issue.severity.upper()}] {issue.description}")
                if issue.suggestion:
                    lines.append(f"  Fix: {issue.suggestion}")

        if result.suggestions:
            lines.append("\n### Suggestions")
            for s in result.suggestions:
                lines.append(f"- {s}")

        return "\n".join(lines)
