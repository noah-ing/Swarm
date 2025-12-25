"""
Dialogue Engine: Multi-agent conversational debate system.

Manages back-and-forth dialogue between agents where they can:
- See what others have said
- Respond to specific messages
- Build on ideas iteratively
- Reach consensus through discussion
"""

import time
import uuid
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


# Role-specific system prompts
ROLE_PROMPTS = {
    "proposer": """You are a solution proposer in a multi-agent debate. You can see what all other agents have said.

Your job:
- Propose clear, specific solutions to the task
- Respond constructively to critiques
- Acknowledge good points from others
- Defend your choices with reasoning when challenged
- Be willing to modify your approach based on feedback

Keep responses focused and under 300 words. Address others by name when responding to their points.""",

    "critic": """You are a critic in a multi-agent debate. You review proposals from other agents.

Your job:
- Identify specific issues, risks, or flaws in proposals
- Ask clarifying questions
- Suggest concrete improvements
- Acknowledge strengths, not just weaknesses
- Be constructive, not dismissive

Keep responses focused and under 300 words. Address proposers by name when critiquing their solutions.""",

    "moderator": """You are the moderator in a multi-agent debate. You facilitate consensus.

Your job:
- Summarize points of agreement
- Identify remaining disagreements
- Propose compromises or hybrid solutions
- Ask agents to clarify or reconsider positions
- Call for final consensus when discussion has converged

When consensus is reached, output the final solution clearly marked with "FINAL SOLUTION:" prefix.
Keep responses focused and under 300 words.""",
}


class DialogueRoom:
    """
    Manages multi-agent conversation.

    Agents take turns speaking, each seeing the full conversation history.
    The conversation flows: proposers → critic → responses → moderator.
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

    def _format_conversation(self) -> str:
        """Format the full conversation for agents to see."""
        if not self.messages:
            return "(No messages yet)"

        lines = []
        for msg in self.messages:
            prefix = f"[{msg.speaker}]"
            if msg.reply_to:
                # Find the message being replied to
                for m in self.messages:
                    if m.id == msg.reply_to:
                        prefix = f"[{msg.speaker} → @{m.speaker}]"
                        break
            lines.append(f"{prefix}: {msg.content}")

        return "\n\n".join(lines)

    def _emit_message(self, msg: DialogueMessage):
        """Emit a dialogue message event for the dashboard."""
        # Get reply-to speaker name if applicable
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

        # Emit event and callback
        self._emit_message(msg)
        if self.on_message:
            self.on_message(msg)

        return msg

    def _get_agent_response(self, participant: DialogueParticipant, prompt: str) -> str:
        """Get a response from an agent given the conversation context."""
        # Build the conversation context
        conversation = self._format_conversation()

        user_content = f"""## Task
{self.task}

## Context
{self.context if self.context else "(None provided)"}

## Conversation So Far
{conversation}

## Your Turn
{prompt}"""

        # Use the agent's chat method with role-specific system prompt
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

    def run_dialogue(self, max_rounds: int = 3, proposer_count: int = 2) -> DialogueResult:
        """
        Run the full dialogue session with turn-taking.

        Flow per round:
        1. Proposers share/update their approaches
        2. Critic responds to proposals
        3. Moderator summarizes and checks for consensus

        Args:
            max_rounds: Maximum conversation rounds
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

        for round_num in range(1, max_rounds + 1):
            # Round header event
            self.event_bus.emit(SwarmEvent(
                type=EventType.THOUGHT,
                agent_name="DialogueRoom",
                content=f"Starting round {round_num}/{max_rounds}",
                data={"round": round_num},
            ))

            # Phase 1: Proposers speak
            for proposer in proposers:
                if round_num == 1:
                    prompt = "Share your initial approach to solving this task. Be specific and concrete."
                    msg_type = "proposal"
                else:
                    prompt = "Given the discussion so far, update or defend your approach. Address any critiques."
                    msg_type = "response"

                content = self._get_agent_response(proposer, prompt)
                self.speak(proposer.name, content, msg_type=msg_type)

            # Phase 2: Critic responds
            for critic in critics:
                if round_num == 1:
                    prompt = "Review the proposals above. Identify issues, ask questions, and suggest improvements."
                else:
                    prompt = "Given the updated proposals and discussion, provide any remaining concerns or acknowledge improvements."

                content = self._get_agent_response(critic, prompt)
                self.speak(critic.name, content, msg_type="critique")

            # Phase 3: Moderator checks for consensus
            moderator = moderators[0]
            if round_num < max_rounds:
                prompt = "Summarize the current state of discussion. Are we converging? What needs resolution?"
            else:
                prompt = "This is the final round. Synthesize the best solution from the discussion. Output 'FINAL SOLUTION:' followed by the solution."

            content = self._get_agent_response(moderator, prompt)
            self.speak(moderator.name, content, msg_type="consensus" if round_num == max_rounds else "summary")

            # Check if consensus reached early
            if "FINAL SOLUTION:" in content:
                final_solution = content.split("FINAL SOLUTION:")[-1].strip()
                consensus_reached = True
                break

        # If no explicit final solution, extract from last moderator message
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
    """Simple agent for dialogue participation (not abstract)."""

    def __init__(self, model: str = "sonnet"):
        from config import get_settings
        import anthropic

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
            "max_tokens": 2048,
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
    moderator_model: str = "sonnet",
    working_dir: Optional[str] = None,
) -> DialogueRoom:
    """
    Create a dialogue room with participants.

    Args:
        task: The task to debate
        context: Additional context
        proposer_models: Models for proposers (default: ["haiku", "sonnet"])
        critic_model: Model for critic
        moderator_model: Model for moderator
        working_dir: Working directory for agents

    Returns:
        Configured DialogueRoom ready for dialogue
    """
    proposer_models = proposer_models or ["haiku", "sonnet"]

    room = DialogueRoom(task=task, context=context)

    # Add proposers
    for i, model in enumerate(proposer_models):
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
