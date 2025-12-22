"""Orchestrator agent for task decomposition and coordination."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import BaseAgent
from .grunt import GruntAgent, GruntResult
from .qa import QAAgent, QAResult
from router import ModelRouter
from config import get_settings


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{name}.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


ORCHESTRATOR_SYSTEM_PROMPT = load_prompt("orchestrator") or """You are a task orchestrator. Your job is to break down complex tasks into smaller, focused subtasks that can be executed independently.

## Your Role

1. Analyze the user's task
2. Decompose it into clear, atomic subtasks
3. Identify dependencies between subtasks
4. Output a structured plan

## Guidelines

- Each subtask should be completable by a focused worker with minimal context
- Subtasks should be specific and actionable
- Include what files might be involved if known
- Order subtasks by dependency (independent tasks first)
- Tasks with no dependencies can run in parallel

## Response Format

Respond with a JSON object:

```json
{
    "analysis": "Brief analysis of the task",
    "subtasks": [
        {
            "id": 1,
            "task": "Clear description of what to do",
            "depends_on": [],
            "files_hint": ["optional list of relevant files"],
            "complexity": "low|medium|high"
        }
    ],
    "completion_criteria": "How to know when the overall task is done"
}
```
"""


@dataclass
class Subtask:
    """A subtask to be executed."""

    id: int
    task: str
    depends_on: list[int] = field(default_factory=list)
    files_hint: list[str] = field(default_factory=list)
    complexity: str = "medium"
    status: str = "pending"  # pending, running, completed, failed
    result: GruntResult | None = None
    qa_result: QAResult | None = None
    retries: int = 0


@dataclass
class OrchestratorResult:
    """Result of orchestration."""

    success: bool
    message: str
    subtasks: list[Subtask] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class Orchestrator(BaseAgent):
    """Decomposes tasks and coordinates grunt execution."""

    def __init__(self, model: str = "sonnet", working_dir: str | None = None):
        super().__init__(model=model, system_prompt=ORCHESTRATOR_SYSTEM_PROMPT)
        self.settings = get_settings()
        self.router = ModelRouter()
        self.subtasks: list[Subtask] = []
        self.working_dir = working_dir
        self.on_subtask_start: callable | None = None
        self.on_subtask_complete: callable | None = None

    def decompose(self, task: str, context: str = "") -> list[Subtask]:
        """
        Decompose a task into subtasks.

        Args:
            task: The main task to decompose
            context: Optional context about the codebase/project

        Returns:
            List of Subtask objects
        """
        content = f"## Task\n\n{task}"
        if context:
            content += f"\n\n## Context\n\n{context}"

        messages = [{"role": "user", "content": content}]
        response = self.chat(messages=messages)

        # Parse JSON response
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

            data = json.loads(text)
            subtasks = []

            for st in data.get("subtasks", []):
                subtasks.append(Subtask(
                    id=st.get("id", len(subtasks) + 1),
                    task=st.get("task", ""),
                    depends_on=st.get("depends_on", []),
                    files_hint=st.get("files_hint", []),
                    complexity=st.get("complexity", "medium"),
                ))

            self.subtasks = subtasks
            return subtasks

        except (json.JSONDecodeError, KeyError):
            # If parsing fails, treat the whole task as one subtask
            self.subtasks = [Subtask(id=1, task=task)]
            return self.subtasks

    def _get_ready_subtasks(self, completed_ids: set[int]) -> list[Subtask]:
        """Get subtasks that are ready to run (dependencies met)."""
        ready = []
        for st in self.subtasks:
            if st.status != "pending":
                continue
            # Check if all dependencies are completed
            if all(dep_id in completed_ids for dep_id in st.depends_on):
                ready.append(st)
        return ready

    def run(
        self,
        task: str,
        context: str = "",
        max_retries: int | None = None,
        skip_qa: bool = False,
        parallel: bool = True,
    ) -> OrchestratorResult:
        """
        Execute a task end-to-end.

        Args:
            task: The task to complete
            context: Optional context
            max_retries: Maximum retries per subtask
            skip_qa: Skip QA validation (faster but less reliable)
            parallel: Run independent subtasks in parallel

        Returns:
            OrchestratorResult with outcomes
        """
        if parallel:
            return asyncio.run(self.run_async(task, context, max_retries, skip_qa))
        else:
            return self._run_sequential(task, context, max_retries, skip_qa)

    def _run_sequential(
        self,
        task: str,
        context: str = "",
        max_retries: int | None = None,
        skip_qa: bool = False,
    ) -> OrchestratorResult:
        """Sequential execution (original behavior)."""
        max_retries = max_retries or self.settings.max_retries

        subtasks = self.decompose(task, context)
        if not subtasks:
            return OrchestratorResult(
                success=False,
                message="Failed to decompose task",
            )

        total_input = self.total_input_tokens
        total_output = self.total_output_tokens
        completed_results: dict[int, str] = {}

        for subtask in subtasks:
            for dep_id in subtask.depends_on:
                if dep_id not in completed_results:
                    subtask.status = "failed"
                    continue

            subtask.status = "running"
            if self.on_subtask_start:
                self.on_subtask_start(subtask)

            subtask_context = context
            if subtask.depends_on:
                subtask_context += "\n\n## Previous Results\n\n"
                for dep_id in subtask.depends_on:
                    if dep_id in completed_results:
                        subtask_context += f"### Subtask {dep_id}\n{completed_results[dep_id]}\n\n"

            model = self.router.select(subtask.task, complexity=subtask.complexity)

            while subtask.retries <= max_retries:
                grunt = GruntAgent(model=model, working_dir=self.working_dir)
                result = grunt.run(subtask.task, subtask_context)
                subtask.result = result
                total_input += result.input_tokens
                total_output += result.output_tokens

                if not result.success:
                    subtask.retries += 1
                    model = self.router.escalate(model)
                    subtask_context += f"\n\n## Previous Attempt Failed\n\n{result.error}"
                    continue

                if not skip_qa:
                    qa = QAAgent()
                    qa_result = qa.run(
                        task=subtask.task,
                        grunt_output=result.result,
                        files_modified=result.files_modified,
                    )
                    subtask.qa_result = qa_result
                    total_input += qa_result.input_tokens
                    total_output += qa_result.output_tokens

                    if not qa_result.approved:
                        subtask.retries += 1
                        model = self.router.escalate(model)
                        subtask_context += f"\n\n## QA Feedback\n\n{qa_result.feedback}"
                        if qa_result.suggestions:
                            subtask_context += "\n\nSuggestions:\n"
                            for s in qa_result.suggestions:
                                subtask_context += f"- {s}\n"
                        continue

                subtask.status = "completed"
                completed_results[subtask.id] = result.result
                if self.on_subtask_complete:
                    self.on_subtask_complete(subtask)
                break

            if subtask.status != "completed":
                subtask.status = "failed"

        return self._build_result(subtasks, total_input, total_output)

    async def run_async(
        self,
        task: str,
        context: str = "",
        max_retries: int | None = None,
        skip_qa: bool = False,
    ) -> OrchestratorResult:
        """
        Execute a task with parallel subtask execution.

        Args:
            task: The task to complete
            context: Optional context
            max_retries: Maximum retries per subtask
            skip_qa: Skip QA validation

        Returns:
            OrchestratorResult with outcomes
        """
        max_retries = max_retries or self.settings.max_retries

        subtasks = self.decompose(task, context)
        if not subtasks:
            return OrchestratorResult(
                success=False,
                message="Failed to decompose task",
            )

        total_input = self.total_input_tokens
        total_output = self.total_output_tokens
        completed_results: dict[int, str] = {}
        completed_ids: set[int] = set()

        # Process in waves until all done or stuck
        while True:
            ready = self._get_ready_subtasks(completed_ids)
            if not ready:
                # Check if we're done or stuck
                pending = [st for st in subtasks if st.status == "pending"]
                if not pending:
                    break
                # Stuck - dependencies can't be met
                for st in pending:
                    st.status = "failed"
                break

            # Execute ready subtasks in parallel
            tasks = []
            for subtask in ready:
                subtask.status = "running"
                if self.on_subtask_start:
                    self.on_subtask_start(subtask)

                # Build context from dependencies
                subtask_context = context
                if subtask.depends_on:
                    subtask_context += "\n\n## Previous Results\n\n"
                    for dep_id in subtask.depends_on:
                        if dep_id in completed_results:
                            subtask_context += f"### Subtask {dep_id}\n{completed_results[dep_id]}\n\n"

                tasks.append(self._execute_subtask(
                    subtask, subtask_context, max_retries, skip_qa
                ))

            # Wait for all parallel tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for subtask, result in zip(ready, results):
                if isinstance(result, Exception):
                    subtask.status = "failed"
                    subtask.result = GruntResult(
                        success=False,
                        result="",
                        error=str(result),
                    )
                else:
                    input_tokens, output_tokens = result
                    total_input += input_tokens
                    total_output += output_tokens

                    if subtask.status == "completed":
                        completed_ids.add(subtask.id)
                        if subtask.result:
                            completed_results[subtask.id] = subtask.result.result
                        if self.on_subtask_complete:
                            self.on_subtask_complete(subtask)

        return self._build_result(subtasks, total_input, total_output)

    async def _execute_subtask(
        self,
        subtask: Subtask,
        context: str,
        max_retries: int,
        skip_qa: bool,
    ) -> tuple[int, int]:
        """Execute a single subtask with retries."""
        model = self.router.select(subtask.task, complexity=subtask.complexity)
        input_tokens = 0
        output_tokens = 0

        while subtask.retries <= max_retries:
            grunt = GruntAgent(model=model, working_dir=self.working_dir)
            result = await grunt.run_async(subtask.task, context)
            subtask.result = result
            input_tokens += result.input_tokens
            output_tokens += result.output_tokens

            if not result.success:
                subtask.retries += 1
                model = self.router.escalate(model)
                context += f"\n\n## Previous Attempt Failed\n\n{result.error}"
                continue

            if not skip_qa:
                qa = QAAgent()
                # QA is sync for now
                qa_result = qa.run(
                    task=subtask.task,
                    grunt_output=result.result,
                    files_modified=result.files_modified,
                )
                subtask.qa_result = qa_result
                input_tokens += qa_result.input_tokens
                output_tokens += qa_result.output_tokens

                if not qa_result.approved:
                    subtask.retries += 1
                    model = self.router.escalate(model)
                    context += f"\n\n## QA Feedback\n\n{qa_result.feedback}"
                    if qa_result.suggestions:
                        context += "\n\nSuggestions:\n"
                        for s in qa_result.suggestions:
                            context += f"- {s}\n"
                    continue

            subtask.status = "completed"
            break

        if subtask.status != "completed":
            subtask.status = "failed"

        return input_tokens, output_tokens

    def _build_result(
        self,
        subtasks: list[Subtask],
        total_input: int,
        total_output: int,
    ) -> OrchestratorResult:
        """Build the final result."""
        all_completed = all(st.status == "completed" for st in subtasks)

        if all_completed:
            summary_parts = [f"Completed all {len(subtasks)} subtasks:"]
            for st in subtasks:
                summary_parts.append(f"\n{st.id}. {st.task[:50]}...")
                if st.result:
                    summary_parts.append(f"   → {st.result.result[:100]}...")

            return OrchestratorResult(
                success=True,
                message="\n".join(summary_parts),
                subtasks=subtasks,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
            )
        else:
            failed = [st for st in subtasks if st.status == "failed"]
            return OrchestratorResult(
                success=False,
                message=f"Failed {len(failed)}/{len(subtasks)} subtasks",
                subtasks=subtasks,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
            )
