"""QA agent for validating grunt outputs."""

from dataclasses import dataclass, field

from .base import BaseAgent


QA_SYSTEM_PROMPT = """You are a quality assurance reviewer. Your job is to validate whether a task was completed correctly.

## Your Role

1. Review the original task/subtask
2. Examine the grunt's output and any files it modified
3. Determine if the task was completed successfully
4. Provide specific, actionable feedback if not

## Evaluation Criteria

- Did the grunt complete what was asked?
- Are there any obvious errors or bugs?
- Did it make unnecessary changes beyond the scope?
- Is the output reasonable and functional?

## Response Format

You must respond with a JSON object:

```json
{
    "approved": true/false,
    "confidence": 0.0-1.0,
    "feedback": "Brief explanation of your decision",
    "suggestions": ["Specific fix 1", "Specific fix 2"]
}
```

Be strict but fair. Approve if the core task is done correctly, even if not perfect.
Reject if there are functional issues or the task wasn't completed.
"""


@dataclass
class QAResult:
    """Result of QA review."""

    approved: bool
    confidence: float
    feedback: str
    suggestions: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class QAAgent(BaseAgent):
    """Reviews and validates grunt outputs."""

    def __init__(self, model: str = "sonnet"):
        super().__init__(model=model, system_prompt=QA_SYSTEM_PROMPT)

    def run(
        self,
        task: str,
        grunt_output: str,
        files_modified: list[str] | None = None,
        file_contents: dict[str, str] | None = None,
    ) -> QAResult:
        """
        Review a grunt's work.

        Args:
            task: The original task given to the grunt
            grunt_output: The grunt's output/summary
            files_modified: List of files the grunt modified
            file_contents: Optional dict of {path: content} for modified files

        Returns:
            QAResult with approval decision
        """
        # Build review prompt
        content = f"""## Original Task

{task}

## Grunt Output

{grunt_output}
"""

        if files_modified:
            content += f"\n## Files Modified\n\n"
            for f in files_modified:
                content += f"- {f}\n"

        if file_contents:
            content += "\n## File Contents\n\n"
            for path, contents in file_contents.items():
                # Truncate long files
                if len(contents) > 2000:
                    contents = contents[:2000] + "\n... (truncated)"
                content += f"### {path}\n\n```\n{contents}\n```\n\n"

        content += "\nReview this work and provide your assessment as JSON."

        messages = [{"role": "user", "content": content}]

        response = self.chat(messages=messages)

        # Parse JSON response
        try:
            import json
            import re

            # Extract JSON from response
            text = response.content or "{}"

            # Try to find JSON block
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    text = json_match.group(0)

            data = json.loads(text)

            return QAResult(
                approved=data.get("approved", False),
                confidence=data.get("confidence", 0.5),
                feedback=data.get("feedback", "No feedback provided"),
                suggestions=data.get("suggestions", []),
                input_tokens=self.total_input_tokens,
                output_tokens=self.total_output_tokens,
            )

        except (json.JSONDecodeError, KeyError) as e:
            # If we can't parse, assume rejection with low confidence
            return QAResult(
                approved=False,
                confidence=0.3,
                feedback=f"Could not parse QA response: {response.content}",
                suggestions=["Retry the task"],
                input_tokens=self.total_input_tokens,
                output_tokens=self.total_output_tokens,
            )
