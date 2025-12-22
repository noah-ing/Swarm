"""Orchestrator agent for task decomposition and coordination."""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .base import BaseAgent, StreamEvent
from .grunt import GruntAgent, GruntResult
from .qa import QAAgent, QAResult
from router import ModelRouter
from config import get_settings
from memory import get_memory_store, Memory


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{name}.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


ORCHESTRATOR_SYSTEM_PROMPT = load_prompt("orchestrator") or """You are a task orchestrator. Analyze tasks and decide whether to run directly or decompose into subtasks."""


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
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class OrchestratorResult:
    """Result of orchestration."""

    success: bool
    message: str
    strategy: str = "direct"  # direct or decompose
    subtasks: list[Subtask] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    duration_seconds: float = 0.0


class Orchestrator(BaseAgent):
    """Decomposes tasks and coordinates grunt execution."""

    def __init__(self, model: str = "sonnet", working_dir: str | None = None):
        super().__init__(model=model, system_prompt=ORCHESTRATOR_SYSTEM_PROMPT)
        self.settings = get_settings()
        self.router = ModelRouter()
        self.memory = get_memory_store()
        self.subtasks: list[Subtask] = []
        self.working_dir = working_dir

        # Callbacks for UI integration
        self.on_subtask_start: Callable[[Subtask], None] | None = None
        self.on_subtask_complete: Callable[[Subtask], None] | None = None
        self.on_subtask_update: Callable[[Subtask], None] | None = None
        self.on_log: Callable[[str], None] | None = None

    def _log(self, message: str):
        """Emit a log message."""
        if self.on_log:
            self.on_log(message)

    def analyze(self, task: str, context: str = "") -> dict:
        """
        Analyze a task and decide execution strategy.

        Returns:
            Dict with 'strategy' ('direct' or 'decompose') and plan details
        """
        # Check memory for similar successful solutions
        similar = self.memory.get_similar_solutions(task, limit=2)

        content = f"## Task\n\n{task}"
        if context:
            content += f"\n\n## Context\n\n{context}"
        if similar:
            content += f"\n\n{similar}"

        content += "\n\nAnalyze this task. Should it run directly (single grunt) or be decomposed? Respond with JSON."

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

            return json.loads(text)

        except (json.JSONDecodeError, KeyError):
            # Default to direct execution on parse failure
            return {"strategy": "direct", "task": task}

    def decompose(self, task: str, context: str = "") -> list[Subtask]:
        """
        Decompose a task into subtasks.

        Returns:
            List of Subtask objects (empty if direct execution)
        """
        analysis = self.analyze(task, context)

        if analysis.get("strategy") == "direct":
            self._log("Strategy: Direct execution (single grunt)")
            return []

        self._log(f"Strategy: Decompose ({len(analysis.get('subtasks', []))} subtasks)")

        subtasks = []
        for st in analysis.get("subtasks", []):
            subtasks.append(Subtask(
                id=st.get("id", len(subtasks) + 1),
                task=st.get("task", ""),
                depends_on=st.get("depends_on", []),
                files_hint=st.get("files_hint", []),
                complexity=st.get("complexity", "medium"),
            ))

        self.subtasks = subtasks
        return subtasks

    def _get_ready_subtasks(self, completed_ids: set[int]) -> list[Subtask]:
        """Get subtasks that are ready to run (dependencies met)."""
        ready = []
        for st in self.subtasks:
            if st.status != "pending":
                continue
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
        stream: bool = False,
    ) -> OrchestratorResult:
        """
        Execute a task end-to-end.

        Args:
            task: The task to complete
            context: Optional context
            max_retries: Maximum retries per subtask
            skip_qa: Skip QA validation
            parallel: Run independent subtasks in parallel
            stream: Enable streaming output

        Returns:
            OrchestratorResult with outcomes
        """
        start_time = time.time()

        # Analyze and decide strategy
        subtasks = self.decompose(task, context)

        if not subtasks:
            # Direct execution - single grunt
            result = self._run_direct(task, context, max_retries, skip_qa, stream)
            result.duration_seconds = time.time() - start_time
            return result

        # Decomposed execution
        if parallel:
            result = asyncio.run(self._run_parallel(
                task, context, subtasks, max_retries, skip_qa, stream
            ))
        else:
            result = self._run_sequential(
                task, context, subtasks, max_retries, skip_qa, stream
            )

        result.duration_seconds = time.time() - start_time
        return result

    def _run_direct(
        self,
        task: str,
        context: str,
        max_retries: int | None,
        skip_qa: bool,
        stream: bool,
    ) -> OrchestratorResult:
        """Run task directly with a single grunt."""
        max_retries = max_retries or self.settings.max_retries

        # Check for similar past solutions
        similar = self.memory.get_similar_solutions(task, limit=2)
        if similar:
            context = f"{context}\n\n{similar}" if context else similar

        model = self.router.select(task)
        self._log(f"Model: {model}")

        grunt = GruntAgent(model=model, working_dir=self.working_dir)

        # Set up streaming if enabled
        if stream and self.on_stream:
            grunt.on_stream = self.on_stream

        retries = 0
        result = None
        qa_result = None

        while retries <= max_retries:
            result = grunt.run(task, context)

            if not result.success:
                retries += 1
                self._log(f"Retry {retries}/{max_retries}: {result.error}")
                model = self.router.escalate(model)
                grunt = GruntAgent(model=model, working_dir=self.working_dir)
                if stream and self.on_stream:
                    grunt.on_stream = self.on_stream
                context += f"\n\n## Previous Attempt Failed\n\n{result.error}"
                continue

            if not skip_qa:
                qa = QAAgent()
                qa_result = qa.run(
                    task=task,
                    grunt_output=result.result,
                    files_modified=result.files_modified,
                )

                if not qa_result.approved:
                    retries += 1
                    self._log(f"QA rejected, retry {retries}/{max_retries}")
                    model = self.router.escalate(model)
                    grunt = GruntAgent(model=model, working_dir=self.working_dir)
                    if stream and self.on_stream:
                        grunt.on_stream = self.on_stream
                    context += f"\n\n## QA Feedback\n\n{qa_result.feedback}"
                    continue

            break

        if result and result.success:
            # Store in memory
            self.memory.store(Memory(
                id="",
                task=task,
                solution=result.result,
                success=True,
                model=model,
                tokens_used=result.input_tokens + result.output_tokens,
                cost_usd=grunt.get_cost(),
                files_modified=result.files_modified,
            ))

            return OrchestratorResult(
                success=True,
                message=result.result,
                strategy="direct",
                total_input_tokens=grunt.total_input_tokens,
                total_output_tokens=grunt.total_output_tokens,
            )
        else:
            return OrchestratorResult(
                success=False,
                message=result.error if result else "Unknown error",
                strategy="direct",
                total_input_tokens=grunt.total_input_tokens if grunt else 0,
                total_output_tokens=grunt.total_output_tokens if grunt else 0,
            )

    def _run_sequential(
        self,
        task: str,
        context: str,
        subtasks: list[Subtask],
        max_retries: int | None,
        skip_qa: bool,
        stream: bool,
    ) -> OrchestratorResult:
        """Sequential execution of subtasks."""
        max_retries = max_retries or self.settings.max_retries

        total_input = self.total_input_tokens
        total_output = self.total_output_tokens
        completed_results: dict[int, str] = {}

        for subtask in subtasks:
            # Check dependencies
            for dep_id in subtask.depends_on:
                if dep_id not in completed_results:
                    subtask.status = "failed"
                    continue

            subtask.status = "running"
            subtask.start_time = time.time()

            if self.on_subtask_start:
                self.on_subtask_start(subtask)

            # Build context from dependencies
            subtask_context = context
            if subtask.depends_on:
                subtask_context += "\n\n## Previous Results\n\n"
                for dep_id in subtask.depends_on:
                    if dep_id in completed_results:
                        subtask_context += f"### Subtask {dep_id}\n{completed_results[dep_id]}\n\n"

            model = self.router.select(subtask.task, complexity=subtask.complexity)

            while subtask.retries <= max_retries:
                grunt = GruntAgent(model=model, working_dir=self.working_dir)

                if stream and self.on_stream:
                    grunt.on_stream = self.on_stream

                result = grunt.run(subtask.task, subtask_context)
                subtask.result = result
                total_input += result.input_tokens
                total_output += result.output_tokens

                if not result.success:
                    subtask.retries += 1
                    model = self.router.escalate(model)
                    subtask_context += f"\n\n## Previous Attempt Failed\n\n{result.error}"
                    if self.on_subtask_update:
                        self.on_subtask_update(subtask)
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
                        if self.on_subtask_update:
                            self.on_subtask_update(subtask)
                        continue

                subtask.status = "completed"
                subtask.end_time = time.time()
                completed_results[subtask.id] = result.result

                if self.on_subtask_complete:
                    self.on_subtask_complete(subtask)
                break

            if subtask.status != "completed":
                subtask.status = "failed"
                subtask.end_time = time.time()

        return self._build_result(subtasks, total_input, total_output)

    async def _run_parallel(
        self,
        task: str,
        context: str,
        subtasks: list[Subtask],
        max_retries: int | None,
        skip_qa: bool,
        stream: bool,
    ) -> OrchestratorResult:
        """Parallel execution of independent subtasks."""
        max_retries = max_retries or self.settings.max_retries
        self.subtasks = subtasks

        total_input = self.total_input_tokens
        total_output = self.total_output_tokens
        completed_results: dict[int, str] = {}
        completed_ids: set[int] = set()

        # Process in waves until all done or stuck
        while True:
            ready = self._get_ready_subtasks(completed_ids)

            if not ready:
                pending = [st for st in subtasks if st.status == "pending"]
                if not pending:
                    break
                # Stuck - mark remaining as failed
                for st in pending:
                    st.status = "failed"
                break

            self._log(f"Wave: {len(ready)} parallel subtasks")

            # Execute ready subtasks in parallel
            tasks = []
            for subtask in ready:
                subtask.status = "running"
                subtask.start_time = time.time()

                if self.on_subtask_start:
                    self.on_subtask_start(subtask)

                subtask_context = context
                if subtask.depends_on:
                    subtask_context += "\n\n## Previous Results\n\n"
                    for dep_id in subtask.depends_on:
                        if dep_id in completed_results:
                            subtask_context += f"### Subtask {dep_id}\n{completed_results[dep_id]}\n\n"

                tasks.append(self._execute_subtask_async(
                    subtask, subtask_context, max_retries, skip_qa, stream
                ))

            # Wait for all parallel tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for subtask, result in zip(ready, results):
                if isinstance(result, Exception):
                    subtask.status = "failed"
                    subtask.end_time = time.time()
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

    async def _execute_subtask_async(
        self,
        subtask: Subtask,
        context: str,
        max_retries: int,
        skip_qa: bool,
        stream: bool,
    ) -> tuple[int, int]:
        """Execute a single subtask with retries."""
        model = self.router.select(subtask.task, complexity=subtask.complexity)
        input_tokens = 0
        output_tokens = 0

        while subtask.retries <= max_retries:
            grunt = GruntAgent(model=model, working_dir=self.working_dir)

            # Note: streaming in parallel mode is complex, skip for now
            result = await grunt.run_async(subtask.task, context)
            subtask.result = result
            input_tokens += result.input_tokens
            output_tokens += result.output_tokens

            if not result.success:
                subtask.retries += 1
                model = self.router.escalate(model)
                context += f"\n\n## Previous Attempt Failed\n\n{result.error}"
                if self.on_subtask_update:
                    self.on_subtask_update(subtask)
                continue

            if not skip_qa:
                qa = QAAgent()
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
                    if self.on_subtask_update:
                        self.on_subtask_update(subtask)
                    continue

            subtask.status = "completed"
            subtask.end_time = time.time()
            break

        if subtask.status != "completed":
            subtask.status = "failed"
            subtask.end_time = time.time()

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
            summary_parts = [f"Completed {len(subtasks)} subtasks:"]
            for st in subtasks:
                summary_parts.append(f"\n**{st.id}. {st.task[:50]}**")
                if st.result:
                    summary_parts.append(f"\n{st.result.result[:200]}")

            # Store successful execution in memory
            self.memory.store(Memory(
                id="",
                task=f"Orchestrated: {len(subtasks)} subtasks",
                solution="\n".join(summary_parts),
                success=True,
                model=self.model,
                tokens_used=total_input + total_output,
                cost_usd=self.get_cost(),
            ))

            return OrchestratorResult(
                success=True,
                message="\n".join(summary_parts),
                strategy="decompose",
                subtasks=subtasks,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
            )
        else:
            failed = [st for st in subtasks if st.status == "failed"]
            return OrchestratorResult(
                success=False,
                message=f"Failed {len(failed)}/{len(subtasks)} subtasks",
                strategy="decompose",
                subtasks=subtasks,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
            )
