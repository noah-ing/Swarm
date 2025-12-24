"""
NegotiatorAgent: Builds consensus between multiple proposed solutions.

Analyzes solutions with their critiques to:
- Select the best approach
- Synthesize hybrid solutions when beneficial
- Provide clear rationale for decisions
"""

import json
from dataclasses import dataclass, field
from typing import Any

from .base import BaseAgent, Message
from .critic import CritiqueResult


@dataclass
class Proposal:
    """A proposed solution with its critique."""
    id: int
    solution: str
    model_used: str
    critique: CritiqueResult | None = None


@dataclass
class ConsensusResult:
    """Result of consensus building."""
    selected_index: int  # Index of selected proposal (-1 if synthesis)
    selected_solution: str  # The chosen/synthesized solution
    rationale: str  # Why this was chosen
    confidence: float  # 0.0-1.0 consensus confidence
    synthesis_used: bool  # Whether we synthesized a new solution
    dissenting_concerns: list[str]  # Unresolved concerns
    input_tokens: int = 0
    output_tokens: int = 0


NEGOTIATOR_SYSTEM_PROMPT = """You are an expert at evaluating multiple solutions and building consensus. Your role is to analyze competing proposals, consider their critiques, and determine the best path forward.

When evaluating solutions:
1. Consider the critique scores and identified issues
2. Weigh trade-offs between different approaches
3. Look for opportunities to synthesize the best parts of multiple solutions
4. Provide clear rationale for your decision

Respond in this exact JSON format:
{
    "selected_index": 0,  // Index of best solution, or -1 if synthesizing
    "rationale": "Clear explanation of why this solution was chosen",
    "synthesis": null,  // If synthesizing: the combined solution, otherwise null
    "confidence": 0.0-1.0,  // How confident in this selection
    "dissenting_concerns": ["Concerns that remain unaddressed"]
}

If you synthesize a solution, selected_index should be -1 and synthesis should contain the combined approach."""


class NegotiatorAgent(BaseAgent):
    """
    Builds consensus between multiple proposed solutions.

    Analyzes trade-offs, considers critiques, and either selects
    the best solution or synthesizes a hybrid approach.
    """

    def __init__(self, model: str = "sonnet"):
        super().__init__(model=model)
        self.system_prompt = NEGOTIATOR_SYSTEM_PROMPT

    def build_consensus(
        self,
        task: str,
        proposals: list[Proposal],
        context: str = "",
    ) -> ConsensusResult:
        """
        Build consensus from multiple proposals.

        Args:
            task: The original task
            proposals: List of proposals with critiques
            context: Additional context

        Returns:
            ConsensusResult with selected/synthesized solution
        """
        user_content = f"## Task\n{task}\n\n## Proposals\n"

        for i, proposal in enumerate(proposals):
            user_content += f"\n### Proposal {i} (Model: {proposal.model_used})\n"
            user_content += f"{proposal.solution}\n"

            if proposal.critique:
                user_content += f"\n**Critique Score:** {proposal.critique.score:.0%}\n"
                user_content += f"**Issues:** {len(proposal.critique.issues)}\n"
                if proposal.critique.issues:
                    for issue in proposal.critique.issues[:3]:
                        user_content += f"- [{issue.severity}] {issue.description}\n"
                if proposal.critique.strengths:
                    user_content += f"**Strengths:** {', '.join(proposal.critique.strengths[:3])}\n"

        if context:
            user_content += f"\n## Context\n{context}"

        user_content += "\n\nAnalyze these proposals and provide your consensus decision in the specified JSON format."

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_content),
        ]

        response = self._call_llm(messages)

        return self._parse_consensus(
            response.content,
            proposals,
            response.input_tokens,
            response.output_tokens,
        )

    def _parse_consensus(
        self,
        content: str,
        proposals: list[Proposal],
        input_tokens: int,
        output_tokens: int,
    ) -> ConsensusResult:
        """Parse the LLM response into a ConsensusResult."""
        try:
            # Extract JSON from response
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            selected_index = data.get("selected_index", 0)
            synthesis = data.get("synthesis")
            synthesis_used = synthesis is not None and selected_index == -1

            # Get the selected solution
            if synthesis_used:
                selected_solution = synthesis
            elif 0 <= selected_index < len(proposals):
                selected_solution = proposals[selected_index].solution
            else:
                selected_solution = proposals[0].solution if proposals else ""
                selected_index = 0

            return ConsensusResult(
                selected_index=selected_index,
                selected_solution=selected_solution,
                rationale=data.get("rationale", "No rationale provided"),
                confidence=float(data.get("confidence", 0.7)),
                synthesis_used=synthesis_used,
                dissenting_concerns=data.get("dissenting_concerns", []),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Default to first proposal if parsing fails
            return ConsensusResult(
                selected_index=0,
                selected_solution=proposals[0].solution if proposals else "",
                rationale="Defaulted to first proposal due to parsing error",
                confidence=0.5,
                synthesis_used=False,
                dissenting_concerns=["Consensus parsing failed"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    def format_consensus(self, result: ConsensusResult) -> str:
        """Format a consensus result for display."""
        lines = [f"## Consensus Decision (Confidence: {result.confidence:.0%})"]

        if result.synthesis_used:
            lines.append("\n**Decision:** Synthesized hybrid solution")
        else:
            lines.append(f"\n**Decision:** Selected Proposal {result.selected_index}")

        lines.append(f"\n### Rationale\n{result.rationale}")

        if result.dissenting_concerns:
            lines.append("\n### Unresolved Concerns")
            for concern in result.dissenting_concerns:
                lines.append(f"- {concern}")

        return "\n".join(lines)
