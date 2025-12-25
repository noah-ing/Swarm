"""
Supervisor: The cognitive layer above orchestration.

The Supervisor is the "CEO" of Swarm:
- Thinks before acting (meta-cognition)
- Learns from every outcome (reflection)
- Adapts strategies based on success (evolution)
- Understands the codebase it's working on
- Can ask for help when uncertain
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from goals import GoalTree

from .base import BaseAgent, StreamEvent
from .thinker import ThinkerAgent, ThinkingResult
from .orchestrator import Orchestrator, OrchestratorResult
from .grunt import GruntAgent, GruntResult
from brain import get_brain, CognitiveArchitecture
from evolution import get_evolution, PromptEvolution
from codebase import get_codebase_analyzer, CodebaseAnalyzer
from memory import get_memory_store
from knowledge import get_knowledge_store
from effects import get_effect_predictor, EffectPrediction
from rollback import get_rollback_manager, RollbackPlan
from negotiation import get_negotiation_coordinator, NegotiationResult
from agentfactory import get_agent_factory, AgentFactory, AgentBlueprint

# Dashboard events (optional - only if dashboard is available)
try:
    from dashboard.events import get_event_bus, EventType
    _event_bus = get_event_bus()
except ImportError:
    _event_bus = None


@dataclass
class SupervisorResult:
    """Result from the supervisor."""
    success: bool
    message: str
    thinking: ThinkingResult | None = None
    execution_result: OrchestratorResult | GruntResult | None = None
    effect_prediction: EffectPrediction | None = None
    rollback_plan_id: str | None = None
    negotiation_result: NegotiationResult | None = None
    goal_tree: GoalTree | None = None
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
        self.knowledge = get_knowledge_store()
        self.effects = get_effect_predictor()
        self.rollback = get_rollback_manager()
        self.negotiation = get_negotiation_coordinator(working_dir)

        # Lazy import to avoid circular dependency
        from goals import get_hierarchical_planner
        self.hierarchical = get_hierarchical_planner(working_dir)
        self.agent_factory = get_agent_factory()

        # Set current project for cross-project knowledge
        if working_dir:
            self.knowledge.set_current_project(working_dir)

        # Callbacks
        self.on_thinking: Callable[[ThinkingResult], None] | None = None
        self.on_ask_user: Callable[[str], str] | None = None
        self.on_strategy: Callable[[str], None] | None = None
        self.on_learning: Callable[[str], None] | None = None
        self.on_effect_prediction: Callable[[EffectPrediction], None] | None = None
        self.on_negotiation: Callable[[NegotiationResult], None] | None = None
        self.on_goal_progress: Callable[[int, int], None] | None = None
        self.on_agent_spawn: Callable[[AgentBlueprint], None] | None = None

    def _emit_event(self, event_type, content: str = "", **kwargs):
        """Emit an event to the dashboard if available."""
        if _event_bus is None:
            return
        from dashboard.events import SwarmEvent
        event = SwarmEvent(
            type=event_type,
            agent_name=kwargs.get("agent_name", "supervisor"),
            content=content,
            **{k: v for k, v in kwargs.items() if k != "agent_name"}
        )
        _event_bus.emit(event)

    def run(
        self,
        task: str,
        context: str = "",
        skip_thinking: bool = False,
        allow_ask: bool = True,
        skip_qa: bool = False,
        stream: bool = False,
    ) -> SupervisorResult:
        """
        Execute a task with full cognitive processing.

        Args:
            task: The task to execute
            context: Additional context for the task
            skip_thinking: Skip the thinking phase for speed
            allow_ask: Allow asking user for clarification
            skip_qa: Skip quality assurance checks
            stream: Stream output to callback

        Returns:
            SupervisorResult with full execution details
        """
        start_time = time.time()
        result = SupervisorResult(success=False, message="")
        task_id = f"task_{int(start_time * 1000)}"

        # Emit task start event
        if _event_bus:
            self._emit_event(EventType.TASK_START, task[:200], task_id=task_id)

        # Phase 1: Understand the codebase
        full_context = self._phase1_understand_codebase(task, context)

        # Phase 2: Think about the task
        thinking, full_context, should_continue = self._phase2_think_about_task(
            task, full_context, skip_thinking, allow_ask, result
        )
        if not should_continue:
            result.duration_seconds = time.time() - start_time
            return result

        # Phase 3: Select strategy and model
        strategy, recommended_model = self._phase3_select_strategy_and_model(
            thinking, task, full_context, result
        )

        # Phase 3.5: Predict effects before execution
        effect_prediction = self._phase3_5_predict_effects(task, allow_ask, result)
        if effect_prediction and effect_prediction.risk_level == "critical" and allow_ask:
            if not self._confirm_high_risk(effect_prediction, result):
                result.message = "Task cancelled due to high risk"
                result.duration_seconds = time.time() - start_time
                return result

        # Phase 4: Get evolved prompt if available
        prompt_variant_id, evolved_prompt = self._phase4_get_evolved_prompt()

        # Phase 4.5: Create rollback plan
        rollback_plan = self._phase4_5_create_rollback_plan(task, effect_prediction, result)

        # Phase 5: Execute (with optional negotiation for high-risk tasks)
        if self.negotiation.should_negotiate(effect_prediction):
            # High-risk: use multi-agent negotiation
            self._phase5_execute_with_negotiation(
                task, full_context, effect_prediction, result
            )
        else:
            # Normal execution
            self._phase5_execute_task(
                strategy, task, full_context, recommended_model,
                evolved_prompt, skip_qa, stream, result
            )

        # Mark rollback plan as executed
        if rollback_plan:
            self.rollback.mark_executed(rollback_plan.id)

        result.duration_seconds = time.time() - start_time

        # Phase 6: Reflect and learn
        self._phase6_reflect_and_learn(
            task, result, recommended_model
        )

        # Phase 7: Update evolution
        self._phase7_update_evolution(
            prompt_variant_id, result
        )

        # Phase 8: Store project-aware memory
        self._phase8_store_knowledge(task, result, recommended_model)

        # Calculate cost
        costs = self.settings.cost_per_million.get(recommended_model, {"input": 0, "output": 0})
        # Rough split: 70% input, 30% output
        input_tokens = int(result.tokens_used * 0.7)
        output_tokens = result.tokens_used - input_tokens
        result.cost_usd = (
            (input_tokens / 1_000_000) * costs["input"] +
            (output_tokens / 1_000_000) * costs["output"]
        )

        # Emit task complete event
        if _event_bus:
            event_type = EventType.TASK_COMPLETE if result.success else EventType.TASK_ERROR
            self._emit_event(
                event_type,
                result.message[:300] if result.message else "Task completed",
                task_id=task_id,
                tokens=result.tokens_used,
                data={"success": result.success, "duration": result.duration_seconds},
            )

        return result

    def _phase1_understand_codebase(self, task: str, context: str) -> str:
        """Phase 1: Understand the codebase and gather cross-project knowledge."""
        codebase_context = ""
        if self.working_dir:
            try:
                codebase_context = self.codebase.get_context_for_task(
                    self.working_dir, task
                )
            except Exception:
                pass  # Codebase analysis is optional

        # Get cross-project knowledge
        cross_project_context = ""
        try:
            matches = self.knowledge.search_cross_project(task, limit=3)
            if matches:
                parts = ["## Cross-Project Knowledge\n"]
                for m in matches:
                    if m.source_project_name:
                        parts.append(f"### From project '{m.source_project_name}' (relevance: {m.relevance_score:.0%})")
                    parts.append(f"**Task:** {m.task[:200]}")
                    parts.append(f"**Solution:** {m.solution[:300]}")
                    parts.append("")
                cross_project_context = "\n".join(parts)
        except Exception:
            pass  # Cross-project knowledge is optional

        full_context = context
        if codebase_context:
            full_context = f"{full_context}\n\n{codebase_context}"
        if cross_project_context:
            full_context = f"{full_context}\n\n{cross_project_context}"

        return full_context

    def _phase2_think_about_task(
        self, 
        task: str, 
        full_context: str, 
        skip_thinking: bool, 
        allow_ask: bool,
        result: SupervisorResult
    ) -> tuple[ThinkingResult | None, str, bool]:
        """
        Phase 2: Think about the task and potentially interact with user.
        
        Returns:
            Tuple of (thinking_result, updated_context, should_continue)
        """
        thinking = None
        should_continue = True
        
        if not skip_thinking:
            thinker = ThinkerAgent(model=self.model)
            thinking = thinker.think(task, full_context)
            result.thinking = thinking

            if self.on_thinking:
                self.on_thinking(thinking)

            # Emit thought event for dashboard
            if _event_bus and thinking:
                self._emit_event(
                    EventType.THOUGHT,
                    f"Understanding: {thinking.understanding}\n\nStrategy: {thinking.strategy_reasoning}",
                    agent_name="thinker",
                )

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
                        should_continue = False
                        return thinking, full_context, should_continue

                    if user_response and user_response.lower() not in ("yes", "y", "proceed"):
                        # User provided clarification
                        full_context += f"\n\n## User Clarification\n{user_response}"
        
        return thinking, full_context, should_continue

    def _phase3_select_strategy_and_model(
        self, 
        thinking: ThinkingResult | None, 
        task: str, 
        full_context: str,
        result: SupervisorResult
    ) -> tuple[str, str]:
        """Phase 3: Select execution strategy and model."""
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

        # Emit strategy event
        if _event_bus:
            self._emit_event(
                EventType.STRATEGY,
                f"Selected strategy: {strategy} with model: {recommended_model}",
            )

        return strategy, recommended_model

    def _phase3_5_predict_effects(
        self,
        task: str,
        allow_ask: bool,
        result: SupervisorResult,
    ) -> EffectPrediction | None:
        """Phase 3.5: Predict effects before execution."""
        if not self.working_dir:
            return None

        try:
            prediction = self.effects.predict_from_task(self.working_dir, task)
            result.effect_prediction = prediction

            if self.on_effect_prediction:
                self.on_effect_prediction(prediction)

            return prediction
        except Exception:
            return None

    def _confirm_high_risk(
        self,
        prediction: EffectPrediction,
        result: SupervisorResult,
    ) -> bool:
        """Ask user to confirm high-risk changes."""
        if not self.on_ask_user:
            return True  # Proceed if no way to ask

        formatted = self.effects.format_prediction(prediction)
        response = self.on_ask_user(
            f"This task has been assessed as CRITICAL risk:\n\n"
            f"{formatted}\n\n"
            f"Do you want to proceed? (yes/no)"
        )

        result.asked_user = True
        result.user_response = response

        if response and response.lower() in ("yes", "y", "proceed"):
            return True
        return False

    def _phase4_get_evolved_prompt(self) -> tuple[int | None, str | None]:
        """Phase 4: Get evolved prompt if available."""
        return self.evolution.get_prompt("grunt")

    def _phase4_5_create_rollback_plan(
        self,
        task: str,
        effect_prediction: EffectPrediction | None,
        result: SupervisorResult,
    ) -> RollbackPlan | None:
        """Phase 4.5: Create rollback plan before execution."""
        if not self.working_dir:
            return None

        try:
            # Determine files to watch based on effect prediction
            files_to_watch = []
            if effect_prediction:
                files_to_watch = effect_prediction.target_files.copy()
                # Also watch directly affected files
                for af in effect_prediction.affected_files[:5]:
                    if af.impact_type == "direct":
                        files_to_watch.append(af.path)

            # If no effect prediction, watch common files
            if not files_to_watch:
                files_to_watch = self._guess_target_files(task)

            if not files_to_watch:
                return None

            plan = self.rollback.create_plan(
                task=task,
                root_path=self.working_dir,
                files_to_watch=files_to_watch,
            )

            result.rollback_plan_id = plan.id
            return plan

        except Exception:
            return None

    def _guess_target_files(self, task: str) -> list[str]:
        """Guess which files might be modified based on task description."""
        task_lower = task.lower()
        guessed = []

        # Common patterns
        if "readme" in task_lower:
            guessed.append("README.md")
        if "config" in task_lower:
            guessed.extend(["config.py", "settings.py"])
        if "test" in task_lower:
            guessed.append("tests/")

        return guessed[:5]

    def _phase5_execute_with_negotiation(
        self,
        task: str,
        context: str,
        effect_prediction: EffectPrediction,
        result: SupervisorResult,
    ) -> None:
        """Phase 5 (negotiation): Use multi-agent debate for high-risk tasks."""
        proposer_count = self.negotiation.get_proposer_count(effect_prediction)

        # Emit negotiation start
        if _event_bus:
            self._emit_event(
                EventType.MESSAGE,
                f"Starting multi-agent negotiation with {proposer_count} proposers",
                agent_name="negotiator",
            )

        negotiation_result = self.negotiation.negotiate(
            task=task,
            context=context,
            proposer_count=proposer_count,
        )

        result.negotiation_result = negotiation_result

        if self.on_negotiation:
            self.on_negotiation(negotiation_result)

        # Emit negotiation results
        if _event_bus and negotiation_result:
            # Emit proposals
            for i, proposal in enumerate(negotiation_result.all_proposals[:3]):
                self._emit_event(
                    EventType.PROPOSAL,
                    proposal.solution[:200] if proposal.solution else f"Proposal {i+1}",
                    agent_name=f"proposer_{i+1}",
                )

            # Emit consensus
            confidence = negotiation_result.consensus.confidence if negotiation_result.consensus else 0.0
            self._emit_event(
                EventType.CONSENSUS,
                f"Consensus reached: {negotiation_result.final_solution[:200]}",
                agent_name="negotiator",
                data={"confidence": confidence},
            )

        # Use the negotiated solution as the result
        result.success = negotiation_result.success
        result.message = negotiation_result.final_solution
        result.tokens_used = negotiation_result.total_tokens

    def _phase5_execute_task(
        self,
        strategy: str,
        task: str,
        full_context: str,
        recommended_model: str,
        evolved_prompt: str | None,
        skip_qa: bool,
        stream: bool,
        result: SupervisorResult
    ) -> None:
        """Phase 5: Execute the task based on selected strategy."""
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
            elif strategy == "hierarchical":
                # Hierarchical goal decomposition
                self._execute_hierarchical(task, full_context, result)
                return  # Already sets result fields
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

    def _phase6_reflect_and_learn(
        self, 
        task: str, 
        result: SupervisorResult, 
        recommended_model: str
    ) -> None:
        """Phase 6: Reflect on execution and generate learnings."""
        reflection = self.brain.reflect(
            task=task,
            outcome=result.message,
            success=result.success,
            model_used=recommended_model,
            tokens_used=result.tokens_used,
            duration_seconds=result.duration_seconds,
        )

        result.learnings = reflection.insights

        # Emit reflection event
        if _event_bus and reflection.insights:
            self._emit_event(
                EventType.REFLECTION,
                f"Insights: {'; '.join(reflection.insights[:3])}",
                agent_name="brain",
                data={"insights": reflection.insights, "confidence": reflection.confidence},
            )

        if self.on_learning and reflection.insights:
            for insight in reflection.insights:
                self.on_learning(insight)

    def _phase7_update_evolution(
        self,
        prompt_variant_id: int | None,
        result: SupervisorResult
    ) -> None:
        """Phase 7: Update evolution system with task outcome."""
        if prompt_variant_id:
            self.evolution.record_outcome(
                prompt_variant_id,
                result.success,
                result.tokens_used,
                result.duration_seconds,
            )

    def _phase8_store_knowledge(
        self,
        task: str,
        result: SupervisorResult,
        model: str,
    ) -> None:
        """Phase 8: Store project-aware memory for cross-project learning."""
        try:
            # Extract files modified from execution result if available
            files_modified = []
            if result.execution_result:
                if isinstance(result.execution_result, GruntResult):
                    # Could parse result for file mentions
                    pass
                elif hasattr(result.execution_result, 'subtask_results'):
                    # Collect from orchestrator subtasks
                    for sr in result.execution_result.subtask_results:
                        if hasattr(sr, 'result') and sr.result:
                            # Simple heuristic: look for file paths in results
                            pass

            self.knowledge.store_memory(
                task=task,
                solution=result.message[:1000] if result.message else "",
                success=result.success,
                model=model,
                tokens_used=result.tokens_used,
                cost_usd=result.cost_usd,
                files_modified=files_modified,
                tags=[result.strategy_used] if result.strategy_used else [],
            )
        except Exception:
            pass  # Knowledge storage is optional, don't fail the task

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

    def _execute_hierarchical(
        self,
        task: str,
        context: str,
        result: SupervisorResult,
    ) -> None:
        """Execute using hierarchical goal decomposition."""
        # Set up callbacks
        if self.on_goal_progress:
            self.hierarchical.on_progress = self.on_goal_progress

        # Plan the hierarchy
        tree = self.hierarchical.plan(task, context)

        # Execute the tree
        tree = self.hierarchical.execute(tree, context, parallel=True)

        # Store the tree
        result.goal_tree = tree

        # Aggregate results
        result.success = tree.is_complete() and tree.root.status.value == "completed"
        result.message = tree.root.result or tree.root.error or "Hierarchical execution complete"
        result.tokens_used = tree.total_tokens

        # Save for future reference
        self.hierarchical.save_tree(tree)

    def _find_dynamic_agent(self, task: str) -> AgentBlueprint | None:
        """Find a suitable dynamic agent for the task."""
        active_agents = self.agent_factory.get_active_agents()
        if not active_agents:
            return None

        task_lower = task.lower()
        best_match = None
        best_score = 0

        for agent in active_agents:
            # Score based on capability match
            score = 0
            for cap in agent.capabilities:
                if cap.lower() in task_lower:
                    score += 1

            # Bonus for success rate
            total = agent.tasks_completed + agent.tasks_failed
            if total > 5:
                success_rate = agent.tasks_completed / total
                score += success_rate

            if score > best_score:
                best_score = score
                best_match = agent

        # Only return if reasonable match
        if best_score >= 1:
            return best_match
        return None

    def _execute_with_dynamic_agent(
        self,
        agent_blueprint: AgentBlueprint,
        task: str,
        context: str,
        stream: bool,
    ) -> dict:
        """Execute task with a dynamically loaded agent."""
        agent_class = self.agent_factory.load_agent(agent_blueprint.id)
        if not agent_class:
            return {"success": False, "error": "Failed to load dynamic agent"}

        start_time = time.time()
        try:
            agent = agent_class(model=self.model)

            # Execute based on agent role
            if agent_blueprint.role.value == "analyzer":
                result = agent.analyze(task, context)
            elif agent_blueprint.role.value == "validator":
                result = agent.validate(task)
            elif agent_blueprint.role.value == "transformer":
                result = agent.transform(task, context)
            elif agent_blueprint.role.value == "specialist":
                result = agent.consult(task, context)
            else:
                result = agent.run(task, context)

            duration = time.time() - start_time
            tokens = result.get("tokens", 0)

            # Record outcome
            self.agent_factory.record_task(
                agent_blueprint.id,
                success=True,
                tokens=tokens,
                duration=duration,
            )

            return {
                "success": True,
                "result": result.get("result") or result.get("analysis") or result.get("advice") or str(result),
                "tokens": tokens,
                "duration": duration,
                "agent_used": agent_blueprint.name,
            }

        except Exception as e:
            duration = time.time() - start_time
            self.agent_factory.record_task(agent_blueprint.id, success=False, tokens=0, duration=duration)
            return {"success": False, "error": str(e)}

    def spawn_agent_for_need(self, task_pattern: str) -> AgentBlueprint | None:
        """
        Spawn a new agent type based on identified need.

        This is self-modifying architecture - the system creates new agents
        when it identifies gaps in its capabilities.
        """
        # Analyze needs
        needs = self.agent_factory.analyze_needs()

        # Find matching need
        for need in needs:
            if need.pattern.lower() in task_pattern.lower() or task_pattern.lower() in need.pattern.lower():
                # Create agent for this need
                name = f"auto_{need.suggested_role.value}_{len(self.agent_factory.get_active_agents())}"
                blueprint = self.agent_factory.create_agent(
                    name=name,
                    role=need.suggested_role,
                    description=f"Auto-generated agent for: {need.pattern}",
                    capabilities=need.suggested_capabilities,
                )

                if self.on_agent_spawn:
                    self.on_agent_spawn(blueprint)

                return blueprint

        return None

    def get_cognitive_stats(self) -> dict[str, Any]:
        """Get stats from all cognitive components."""
        return {
            "brain": self.brain.get_stats(),
            "memory": self.memory.get_stats(),
            "evolution": self.evolution.get_stats(),
            "knowledge": self.knowledge.get_stats(),
            "rollback": self.rollback.get_stats(),
            "negotiation": self.negotiation.get_stats(),
            "hierarchical": self.hierarchical.get_stats(),
            "agent_factory": self.agent_factory.get_stats(),
        }

    def execute_rollback(self, plan_id: str) -> dict[str, Any]:
        """
        Execute a rollback to restore files to previous state.

        Args:
            plan_id: The rollback plan ID (from SupervisorResult.rollback_plan_id)

        Returns:
            Dict with success status and details
        """
        result = self.rollback.rollback(plan_id)
        return {
            "success": result.success,
            "files_restored": result.files_restored,
            "files_deleted": result.files_deleted,
            "errors": result.errors,
            "used_git": result.used_git,
        }

    def get_recent_rollback_plans(self, limit: int = 5) -> list[dict]:
        """Get recent rollback plans for review."""
        return self.rollback.get_recent_plans(limit)

    def __repr__(self) -> str:
        """String representation."""
        return f"Supervisor(model={self.model}, working_dir={self.working_dir})"