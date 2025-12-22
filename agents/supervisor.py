"""
Supervisor: The cognitive layer above orchestration.

The Supervisor is the "CEO" of Swarm:
- Thinks before acting (meta-cognition)
- Learns from every outcome (reflection)
- Adapts strategies based on success (evolution)
- Understands the codebase it's working on
- Can ask for help when uncertain
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import BaseAgent, StreamEvent
from .thinker import ThinkerAgent, ThinkingResult
from .orchestrator import Orchestrator, OrchestratorResult
from .grunt import GruntAgent, GruntResult
from brain import get_brain, CognitiveArchitecture
from evolution import get_evolution, PromptEvolution
from codebase import get_codebase_analyzer, CodebaseAnalyzer
from memory import get_memory_store


@dataclass
class SupervisorResult:
    """Result from the supervisor."""
    success: bool
    message: str
    thinking: ThinkingResult | None = None
    execution_result: OrchestratorResult | GruntResult | None = None
    strategy_used: str = ""
    model_used: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    learnings: list[str] = field(default_factory=list)
    asked_user: bool = False
    user_response: str | None = None


class Supervisor(BaseAgent):
    """
    The cognitive supervisor that coordinates all of Swarm's intelligence.

    Flow:
    1. Analyze codebase context
    2. Think about the task (meta-cognition)
    3. Check uncertainty - maybe ask user
    4. Select strategy and model
    5. Execute with orchestrator or direct grunt
    6. Reflect on outcome
    7. Update evolution based on results
    """

    def __init__(self, working_dir: str | None = None, model: str = "sonnet"):
        super().__init__(model=model)
        self.working_dir = working_dir
        self.brain = get_brain()
        self.evolution = get_evolution()
        self.codebase = get_codebase_analyzer()
        self.memory = get_memory_store()

        # Callbacks
        self.on_thinking: Callable[[ThinkingResult], None] | None = None
        self.on_ask_user: Callable[[str], str] | None = None
        self.on_strategy: Callable[[str], None] | None = None
        self.on_learning: Callable[[str], None] | None = None

    def run(
        self,
        task: str,
        context: str = "",
        allow_ask: bool = True,
        skip_thinking: bool = False,
        skip_qa: bool = False,
        stream: bool = False,
    ) -> SupervisorResult:
        """
        Execute a task with full cognitive capabilities.

        Args:
            task: The task to execute
            context: Additional context
            allow_ask: Allow asking user for clarification
            skip_thinking: Skip the thinking phase (faster but less smart)
            skip_qa: Skip QA validation
            stream: Enable streaming output

        Returns:
            SupervisorResult with full execution details
        """
        start_time = time.time()
        result = SupervisorResult(success=False, message="")

        # Phase 1: Understand the codebase
        codebase_context = ""
        if self.working_dir:
            try:
                codebase_context = self.codebase.get_context_for_task(
                    self.working_dir, task
                )
            except Exception:
                pass  # Codebase analysis is optional

        full_context = f"{context}\n\n{codebase_context}" if codebase_context else context

        # Phase 2: Think about the task
        thinking = None
        if not skip_thinking:
            thinker = ThinkerAgent(model=self.model)
            thinking = thinker.think(task, full_context)
            result.thinking = thinking

            if self.on_thinking:
                self.on_thinking(thinking)

            # Check if we should ask user
            if allow_ask:
                should_ask, reason = thinker.should_ask_user(thinking)
                if should_ask and self.on_ask_user:
                    user_response = self.on_ask_user(
                        f"I'm uncertain about this task. {reason}\n\n"
                        f"Task: {task}\n\n"
                        f"My understanding: {thinking.understanding}\n\n"
                        f"Should I proceed? (yes/no/clarify)"
                    )
                    result.asked_user = True
                    result.user_response = user_response

                    if user_response and "no" in user_response.lower():
                        result.message = "Task cancelled by user"
                        result.duration_seconds = time.time() - start_time
                        return result

                    if user_response and user_response.lower() not in ("yes", "y", "proceed"):
                        # User provided clarification
                        full_context += f"\n\n## User Clarification\n{user_response}"

        # Phase 3: Select strategy and model
        if thinking:
            strategy = thinking.strategy
            recommended_model = thinking.recommended_model
        else:
            # Use brain's quick assessment
            brain_strategy = self.brain.select_strategy(task, full_context)
            strategy = brain_strategy.name.lower().replace(" ", "_")
            recommended_model = self.brain.get_recommended_model(task, self.model)

        result.strategy_used = strategy
        result.model_used = recommended_model

        if self.on_strategy:
            self.on_strategy(f"Strategy: {strategy}, Model: {recommended_model}")

        # Phase 4: Get evolved prompt if available
        prompt_variant_id, evolved_prompt = self.evolution.get_prompt("grunt")

        # Phase 5: Execute based on strategy
        try:
            if strategy in ("direct", "direct_execution", "template_match"):
                # Single grunt execution
                exec_result = self._execute_direct(
                    task, full_context, recommended_model,
                    evolved_prompt, skip_qa, stream
                )
            elif strategy in ("explore_first", "trial_and_error"):
                # Explore then execute
                exec_result = self._execute_exploratory(
                    task, full_context, recommended_model,
                    skip_qa, stream
                )
            else:
                # Orchestrated execution
                exec_result = self._execute_orchestrated(
                    task, full_context, recommended_model,
                    skip_qa, stream, parallel="parallel" in strategy
                )

            result.execution_result = exec_result

            if isinstance(exec_result, GruntResult):
                result.success = exec_result.success
                result.message = exec_result.result if exec_result.success else (exec_result.error or "Failed")
                result.tokens_used = exec_result.input_tokens + exec_result.output_tokens
            else:
                result.success = exec_result.success
                result.message = exec_result.message
                result.tokens_used = exec_result.total_input_tokens + exec_result.total_output_tokens

        except Exception as e:
            result.success = False
            result.message = f"Execution error: {str(e)}"

        result.duration_seconds = time.time() - start_time

        # Phase 6: Reflect and learn
        reflection = self.brain.reflect(
            task=task,
            outcome=result.message,
            success=result.success,
            model_used=recommended_model,
            tokens_used=result.tokens_used,
            duration_seconds=result.duration_seconds,
        )

        result.learnings = reflection.insights

        if self.on_learning and reflection.insights:
            for insight in reflection.insights:
                self.on_learning(insight)

        # Phase 7: Update evolution
        if prompt_variant_id:
            self.evolution.record_outcome(
                prompt_variant_id,
                result.success,
                result.tokens_used,
                result.duration_seconds,
            )

        # Calculate cost
        costs = self.settings.cost_per_million.get(recommended_model, {"input": 0, "output": 0})
        # Rough split: 70% input, 30% output
        input_tokens = int(result.tokens_used * 0.7)
        output_tokens = result.tokens_used - input_tokens
        result.cost_usd = (
            (input_tokens / 1_000_000) * costs["input"] +
            (output_tokens / 1_000_000) * costs["output"]
        )

        return result

    def _execute_direct(
        self,
        task: str,
        context: str,
        model: str,
        evolved_prompt: str | None,
        skip_qa: bool,
        stream: bool,
    ) -> GruntResult:
        """Execute with a single grunt."""
        grunt = GruntAgent(model=model, working_dir=self.working_dir)

        if evolved_prompt:
            grunt.system_prompt = evolved_prompt

        if stream and self.on_stream:
            grunt.on_stream = self.on_stream

        # Get similar past solutions
        similar = self.memory.get_similar_solutions(task, limit=2)
        if similar:
            context = f"{context}\n\n{similar}"

        return grunt.run(task, context)

    def _execute_exploratory(
        self,
        task: str,
        context: str,
        model: str,
        skip_qa: bool,
        stream: bool,
    ) -> OrchestratorResult:
        """Execute with exploration first."""
        # First, explore
        explore_grunt = GruntAgent(model=model, working_dir=self.working_dir)
        explore_result = explore_grunt.run(
            f"Explore and understand: {task}\n\nDon't make changes yet. "
            f"Just gather information about what exists and what needs to be done.",
            context
        )

        # Then execute with exploration context
        if explore_result.success:
            context = f"{context}\n\n## Exploration Results\n{explore_result.result}"

        orchestrator = Orchestrator(model=model, working_dir=self.working_dir)
        if stream and self.on_stream:
            orchestrator.on_stream = self.on_stream

        return orchestrator.run(task, context=context, skip_qa=skip_qa)

    def _execute_orchestrated(
        self,
        task: str,
        context: str,
        model: str,
        skip_qa: bool,
        stream: bool,
        parallel: bool = True,
    ) -> OrchestratorResult:
        """Execute with full orchestration."""
        orchestrator = Orchestrator(model=model, working_dir=self.working_dir)

        if stream and self.on_stream:
            orchestrator.on_stream = self.on_stream

        return orchestrator.run(
            task,
            context=context,
            skip_qa=skip_qa,
            parallel=parallel,
            stream=stream,
        )

    def get_cognitive_stats(self) -> dict[str, Any]:
        """Get statistics about the cognitive systems."""
        return {
            "brain": self.brain.get_stats(),
            "evolution": self.evolution.get_stats(),
            "memory": self.memory.get_stats(),
        }
