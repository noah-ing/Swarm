"""
Dialogue Engine: Multi-agent conversational debate system.

Manages back-and-forth dialogue between agents where they can:
- See what others have said
- Respond to specific messages
- Build on ideas iteratively
- Reach consensus through discussion

Optimizations:
- Parallel API calls for round 1 proposers
- Early termination on consensus
- Summarized history for later rounds
- 2 rounds default (sufficient for most tasks)
"""

import time
import uuid
import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent

from dashboard.events import get_event_bus, EventType, SwarmEvent


@dataclass
class DialogueMessage:
    """A single message in the agent dialogue."""
    id: str
    speaker: str           # Agent name (e.g., "Proposer-Haiku", "Critic")
    speaker_model: str     # Model used (e.g., "haiku", "sonnet")
    role: str              # "proposer", "critic", "moderator"
    content: str           # The message text
    timestamp: float
    reply_to: Optional[str] = None  # ID of message being replied to
    msg_type: str = "message"  # "proposal", "critique", "response", "consensus"


@dataclass
class DialogueResult:
    """Result of a dialogue session."""
    messages: list[DialogueMessage]
    final_solution: str
    rounds: int
    consensus_reached: bool
    total_tokens: int
    dialogue_id: str


@dataclass
class DialogueParticipant:
    """A participant in the dialogue."""
    name: str
    role: str  # "proposer", "critic", "moderator"
    model: str
    agent: any  # DialogueAgent or BaseAgent
    tokens_used: int = 0


# Role-specific system prompts - kept concise to save tokens
ROLE_PROMPTS = {
    "proposer": """You propose solutions in a multi-agent debate.
- Be specific and concrete
- Respond to critiques constructively
- Keep responses under 250 words""",

    "critic": """You critique proposals in a multi-agent debate.
- Identify specific issues or risks
- Suggest improvements
- Acknowledge strengths
- Keep responses under 250 words""",

    "moderator": """You moderate a multi-agent debate.
- Summarize agreement/disagreement
- Synthesize best approaches
- When ready, output "FINAL SOLUTION:" followed by the solution
- Keep responses under 250 words""",
}


class DialogueRoom:
    """
    Manages multi-agent conversation with optimizations.

    Optimizations:
    - Round 1: Parallel API calls (proposers don't need to see each other)
    - Round 2+: Sequential with summarized history
    - Early exit on consensus
    """

    def __init__(
        self,
        task: str,
        context: str = "",
        on_message: Optional[Callable[[DialogueMessage], None]] = None,
    ):
        self.task = task
        self.context = context
        self.dialogue_id = f"dlg_{uuid.uuid4().hex[:8]}"
        self.messages: list[DialogueMessage] = []
        self.participants: dict[str, DialogueParticipant] = {}
        self.on_message = on_message
        self.event_bus = get_event_bus()
        self.total_tokens = 0

    def add_participant(self, name: str, role: str, model: str, agent):
        """Add an agent to the dialogue."""
        self.participants[name] = DialogueParticipant(
            name=name,
            role=role,
            model=model,
            agent=agent,
        )

    def _summarize_round(self, round_messages: list[DialogueMessage]) -> str:
        """Summarize a round's messages to save tokens."""
        if not round_messages:
            return ""

        summaries = []
        for msg in round_messages:
            # Truncate long messages
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            summaries.append(f"[{msg.speaker}]: {content}")

        return "\n".join(summaries)

    def _format_conversation(self, summarize_before_round: int = 0) -> str:
        """Format conversation, optionally summarizing older rounds."""
        if not self.messages:
            return "(No messages yet)"

        # Group messages by round (roughly 3-5 messages per round)
        if summarize_before_round > 1 and len(self.messages) > 5:
            # Summarize older messages, keep recent ones in full
            msgs_per_round = 5  # approximate
            cutoff = (summarize_before_round - 1) * msgs_per_round
            cutoff = min(cutoff, len(self.messages) - 3)  # Keep at least last 3

            if cutoff > 0:
                old_msgs = self.messages[:cutoff]
                recent_msgs = self.messages[cutoff:]

                summary = "[Previous discussion summary]\n"
                summary += self._summarize_round(old_msgs)
                summary += "\n\n[Recent messages]\n"

                lines = []
                for msg in recent_msgs:
                    lines.append(f"[{msg.speaker}]: {msg.content}")
                summary += "\n\n".join(lines)

                return summary

        # Full conversation
        lines = []
        for msg in self.messages:
            prefix = f"[{msg.speaker}]"
            if msg.reply_to:
                for m in self.messages:
                    if m.id == msg.reply_to:
                        prefix = f"[{msg.speaker} → @{m.speaker}]"
                        break
            lines.append(f"{prefix}: {msg.content}")

        return "\n\n".join(lines)

    def _emit_message(self, msg: DialogueMessage):
        """Emit a dialogue message event for the dashboard."""
        reply_to_speaker = ""
        if msg.reply_to:
            for m in self.messages:
                if m.id == msg.reply_to:
                    reply_to_speaker = m.speaker
                    break

        self.event_bus.emit(SwarmEvent(
            type=EventType.DIALOGUE,
            agent_id=self.dialogue_id,
            agent_name=msg.speaker,
            agent_role=msg.role,
            target_agent_name=reply_to_speaker,
            content=msg.content,
            model=msg.speaker_model,
            data={
                "msg_type": msg.msg_type,
                "dialogue_id": self.dialogue_id,
                "reply_to_id": msg.reply_to,
            },
        ))

    def speak(
        self,
        speaker_name: str,
        content: str,
        msg_type: str = "message",
        reply_to: Optional[str] = None,
    ) -> DialogueMessage:
        """Agent sends a message to the room."""
        participant = self.participants.get(speaker_name)
        if not participant:
            raise ValueError(f"Unknown speaker: {speaker_name}")

        msg = DialogueMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            speaker=speaker_name,
            speaker_model=participant.model,
            role=participant.role,
            content=content,
            timestamp=time.time(),
            reply_to=reply_to,
            msg_type=msg_type,
        )
        self.messages.append(msg)

        self._emit_message(msg)
        if self.on_message:
            self.on_message(msg)

        return msg

    def _get_agent_response(
        self,
        participant: DialogueParticipant,
        prompt: str,
        current_round: int = 1,
    ) -> str:
        """Get a response from an agent given the conversation context."""
        # Summarize older rounds to save tokens
        conversation = self._format_conversation(summarize_before_round=current_round)

        user_content = f"""## Task
{self.task}

## Context
{self.context if self.context else "(None)"}

## Discussion
{conversation}

## Your Turn
{prompt}"""

        original_prompt = participant.agent.system_prompt
        participant.agent.system_prompt = ROLE_PROMPTS.get(participant.role, "")

        try:
            response = participant.agent.chat(
                messages=[{"role": "user", "content": user_content}],
            )
            participant.tokens_used += response.input_tokens + response.output_tokens
            self.total_tokens += response.input_tokens + response.output_tokens
            return response.content or ""
        finally:
            participant.agent.system_prompt = original_prompt

    def _get_parallel_responses(
        self,
        participants: list[DialogueParticipant],
        prompt: str,
        current_round: int = 1,
    ) -> list[tuple[DialogueParticipant, str]]:
        """Get responses from multiple agents in parallel."""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(participants)) as executor:
            future_to_participant = {
                executor.submit(self._get_agent_response, p, prompt, current_round): p
                for p in participants
            }

            for future in concurrent.futures.as_completed(future_to_participant):
                participant = future_to_participant[future]
                try:
                    content = future.result()
                    results.append((participant, content))
                except Exception as e:
                    results.append((participant, f"Error: {e}"))

        # Sort by original order
        participant_order = {p.name: i for i, p in enumerate(participants)}
        results.sort(key=lambda x: participant_order.get(x[0].name, 999))

        return results

    def run_dialogue(self, max_rounds: int = 2, proposer_count: int = 2) -> DialogueResult:
        """
        Run dialogue with optimizations.

        Optimizations applied:
        - Round 1: Parallel proposer calls
        - Early exit on "FINAL SOLUTION:"
        - Summarized history for round 2+

        Args:
            max_rounds: Maximum rounds (default 2, sufficient for most)
            proposer_count: Number of proposers (2-3)

        Returns:
            DialogueResult with messages and final solution
        """
        proposers = [p for p in self.participants.values() if p.role == "proposer"][:proposer_count]
        critics = [p for p in self.participants.values() if p.role == "critic"]
        moderators = [p for p in self.participants.values() if p.role == "moderator"]

        if not proposers or not critics or not moderators:
            raise ValueError("Dialogue requires at least 1 proposer, 1 critic, and 1 moderator")

        final_solution = ""
        consensus_reached = False
        round_num = 0

        for round_num in range(1, max_rounds + 1):
            self.event_bus.emit(SwarmEvent(
                type=EventType.THOUGHT,
                agent_name="DialogueRoom",
                content=f"Round {round_num}/{max_rounds}",
                data={"round": round_num},
            ))

            # Phase 1: Proposers speak
            if round_num == 1:
                # OPTIMIZATION: Parallel calls for round 1
                prompt = "Propose your approach to this task. Be specific and concrete."
                responses = self._get_parallel_responses(proposers, prompt, round_num)
                for participant, content in responses:
                    self.speak(participant.name, content, msg_type="proposal")
            else:
                # Sequential for later rounds (need to see each other's updates)
                for proposer in proposers:
                    prompt = "Update or defend your approach based on the discussion. Address critiques."
                    content = self._get_agent_response(proposer, prompt, round_num)
                    self.speak(proposer.name, content, msg_type="response")

            # Phase 2: Critic responds
            critic = critics[0]
            if round_num == 1:
                prompt = "Review the proposals. Identify issues and suggest improvements."
            else:
                prompt = "Any remaining concerns? Acknowledge improvements if proposals addressed your critiques."

            content = self._get_agent_response(critic, prompt, round_num)
            self.speak(critic.name, content, msg_type="critique")

            # Phase 3: Moderator
            moderator = moderators[0]
            # Always allow early consensus
            prompt = "Summarize the discussion. If there's clear convergence, output 'FINAL SOLUTION:' followed by the synthesized solution. Otherwise, note what needs resolution."

            content = self._get_agent_response(moderator, prompt, round_num)
            self.speak(moderator.name, content, msg_type="summary")

            # OPTIMIZATION: Early exit on consensus
            if "FINAL SOLUTION:" in content:
                final_solution = content.split("FINAL SOLUTION:")[-1].strip()
                consensus_reached = True
                break

        # Extract solution if not explicit
        if not final_solution:
            for msg in reversed(self.messages):
                if msg.role == "moderator":
                    final_solution = msg.content
                    break

        return DialogueResult(
            messages=self.messages,
            final_solution=final_solution,
            rounds=round_num,
            consensus_reached=consensus_reached,
            total_tokens=self.total_tokens,
            dialogue_id=self.dialogue_id,
        )


class DialogueAgent:
    """Simple agent for dialogue participation."""

    def __init__(self, model: str = "sonnet"):
        from config import get_settings

        self.settings = get_settings()
        self.model = model
        self.system_prompt = ""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            api_key = self.settings.anthropic_api_key
            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key.get_secret_value())
        return self._client

    def chat(self, messages: list[dict], tools: list[dict] = None, model: str = None):
        """Simple chat method for dialogue."""
        from dataclasses import dataclass

        @dataclass
        class Response:
            content: str
            input_tokens: int
            output_tokens: int

        model = model or self.model
        model_id = self.settings.get_model_id(model)

        kwargs = {
            "model": model_id,
            "max_tokens": 1024,  # Reduced from 2048 - dialogues should be concise
            "messages": messages,
        }
        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        response = self.client.messages.create(**kwargs)

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        content = ""
        for block in response.content:
            if block.type == "text":
                content = block.text

        return Response(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def create_dialogue_room(
    task: str,
    context: str = "",
    proposer_models: list[str] = None,
    critic_model: str = "sonnet",
    moderator_model: str = "haiku",  # Haiku sufficient for moderation
    working_dir: Optional[str] = None,
) -> DialogueRoom:
    """
    Create a dialogue room with participants.

    Args:
        task: The task to debate
        context: Additional context
        proposer_models: Models for proposers (default: ["haiku", "sonnet"])
        critic_model: Model for critic
        moderator_model: Model for moderator (default: haiku - cheaper)
        working_dir: Working directory for agents

    Returns:
        Configured DialogueRoom ready for dialogue
    """
    proposer_models = proposer_models or ["haiku", "sonnet"]

    room = DialogueRoom(task=task, context=context)

    # Add proposers
    for model in proposer_models:
        agent = DialogueAgent(model=model)
        room.add_participant(
            name=f"Proposer-{model.title()}",
            role="proposer",
            model=model,
            agent=agent,
        )

    # Add critic
    critic_agent = DialogueAgent(model=critic_model)
    room.add_participant(
        name="Critic",
        role="critic",
        model=critic_model,
        agent=critic_agent,
    )

    # Add moderator
    moderator_agent = DialogueAgent(model=moderator_model)
    room.add_participant(
        name="Moderator",
        role="moderator",
        model=moderator_model,
        agent=moderator_agent,
    )

    return room
