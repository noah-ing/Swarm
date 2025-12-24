"""
Negotiation Coordinator: Orchestrates multi-agent debate.

Manages the full negotiation process:
1. Generate proposals from multiple agents
2. Have each critiqued
3. Build consensus
4. Track negotiation history
"""

import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from agents.grunt import GruntAgent, GruntResult
from agents.critic import CriticAgent, CritiqueResult
from agents.negotiator import NegotiatorAgent, ConsensusResult, Proposal
from effects import EffectPrediction


@dataclass
class NegotiationResult:
    """Result of a full negotiation process."""
    success: bool
    final_solution: str
    proposals_count: int
    rounds: int
    consensus: ConsensusResult | None
    all_proposals: list[Proposal]
    total_tokens: int
    negotiation_id: str


class NegotiationCoordinator:
    """
    Coordinates multi-agent negotiation for improved solution quality.

    Flow:
    1. Generate N proposals from different models/prompts
    2. Critique each proposal
    3. Build consensus or synthesize
    4. Return best solution with rationale
    """

    # Models to use for different proposer slots
    PROPOSER_MODELS = ["haiku", "sonnet", "opus"]

    def __init__(self, working_dir: Optional[str] = None, db_path: Optional[str] = None):
        self.working_dir = working_dir

        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "negotiation.db")

        self.db_path = db_path
        self._init_db()

        # Callbacks
        self.on_proposal: Callable[[int, str], None] | None = None
        self.on_critique: Callable[[int, CritiqueResult], None] | None = None
        self.on_consensus: Callable[[ConsensusResult], None] | None = None

    def _init_db(self):
        """Initialize negotiation tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS negotiations (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                proposer_count INTEGER,
                rounds INTEGER,
                consensus_confidence REAL,
                synthesis_used INTEGER,
                selected_model TEXT,
                total_tokens INTEGER,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_neg_created
            ON negotiations(created_at DESC)
        """)

        conn.commit()
        conn.close()

    def get_proposer_count(self, effect_prediction: Optional[EffectPrediction]) -> int:
        """Determine number of proposers based on risk level."""
        if effect_prediction is None:
            return 0

        risk_to_proposers = {
            "low": 0,
            "medium": 0,
            "high": 2,
            "critical": 3,
        }
        return risk_to_proposers.get(effect_prediction.risk_level, 0)

    def should_negotiate(self, effect_prediction: Optional[EffectPrediction]) -> bool:
        """Check if negotiation should be used."""
        return self.get_proposer_count(effect_prediction) > 0

    def negotiate(
        self,
        task: str,
        context: str = "",
        proposer_count: int = 2,
        max_rounds: int = 1,
    ) -> NegotiationResult:
        """
        Run a full negotiation process.

        Args:
            task: The task to solve
            context: Additional context
            proposer_count: Number of proposals to generate (2-3)
            max_rounds: Maximum debate rounds

        Returns:
            NegotiationResult with final solution
        """
        negotiation_id = f"neg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        proposer_count = min(max(proposer_count, 2), 3)

        # Phase 1: Generate proposals from different models
        proposals = self._generate_proposals(task, context, proposer_count)

        # Phase 2: Critique each proposal
        critic = CriticAgent(model="sonnet")
        for proposal in proposals:
            critique = critic.critique(
                task=task,
                proposed_solution=proposal.solution,
                context=context,
            )
            proposal.critique = critique

            if self.on_critique:
                self.on_critique(proposal.id, critique)

        # Phase 3: Build consensus
        negotiator = NegotiatorAgent(model="sonnet")
        consensus = negotiator.build_consensus(
            task=task,
            proposals=proposals,
            context=context,
        )

        if self.on_consensus:
            self.on_consensus(consensus)

        # Calculate total tokens
        total_tokens = sum(
            (p.critique.input_tokens + p.critique.output_tokens)
            for p in proposals if p.critique
        )
        total_tokens += consensus.input_tokens + consensus.output_tokens

        # Store negotiation record
        self._store_negotiation(
            negotiation_id=negotiation_id,
            task=task,
            proposer_count=proposer_count,
            rounds=1,
            consensus=consensus,
            total_tokens=total_tokens,
        )

        return NegotiationResult(
            success=True,
            final_solution=consensus.selected_solution,
            proposals_count=proposer_count,
            rounds=1,
            consensus=consensus,
            all_proposals=proposals,
            total_tokens=total_tokens,
            negotiation_id=negotiation_id,
        )

    def _generate_proposals(
        self,
        task: str,
        context: str,
        count: int,
    ) -> list[Proposal]:
        """Generate proposals from different models."""
        proposals = []

        for i in range(count):
            model = self.PROPOSER_MODELS[i % len(self.PROPOSER_MODELS)]

            grunt = GruntAgent(model=model, working_dir=self.working_dir)
            result = grunt.run(task, context)

            proposal = Proposal(
                id=i,
                solution=result.result if result.success else f"Failed: {result.error}",
                model_used=model,
            )
            proposals.append(proposal)

            if self.on_proposal:
                self.on_proposal(i, model)

        return proposals

    def _store_negotiation(
        self,
        negotiation_id: str,
        task: str,
        proposer_count: int,
        rounds: int,
        consensus: ConsensusResult,
        total_tokens: int,
    ):
        """Store negotiation record for analytics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO negotiations
            (id, task, proposer_count, rounds, consensus_confidence,
             synthesis_used, selected_model, total_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            negotiation_id,
            task[:500],
            proposer_count,
            rounds,
            consensus.confidence,
            1 if consensus.synthesis_used else 0,
            "synthesis" if consensus.synthesis_used else f"proposal_{consensus.selected_index}",
            total_tokens,
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

    def get_stats(self) -> dict:
        """Get negotiation statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM negotiations")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(consensus_confidence) FROM negotiations")
        avg_confidence = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM negotiations WHERE synthesis_used = 1")
        synthesis_count = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(total_tokens) FROM negotiations")
        avg_tokens = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_negotiations": total,
            "avg_confidence": avg_confidence,
            "synthesis_rate": synthesis_count / total if total > 0 else 0,
            "avg_tokens_per_negotiation": avg_tokens,
        }

    def format_result(self, result: NegotiationResult) -> str:
        """Format negotiation result for display."""
        lines = [
            "## Negotiation Complete",
            f"Proposals: {result.proposals_count}",
            f"Rounds: {result.rounds}",
        ]

        if result.consensus:
            lines.append(f"Confidence: {result.consensus.confidence:.0%}")
            if result.consensus.synthesis_used:
                lines.append("Decision: Synthesized hybrid solution")
            else:
                lines.append(f"Decision: Selected proposal {result.consensus.selected_index}")

            lines.append(f"\n### Rationale\n{result.consensus.rationale}")

        return "\n".join(lines)


# Global instance
_coordinator: Optional[NegotiationCoordinator] = None


def get_negotiation_coordinator(working_dir: Optional[str] = None) -> NegotiationCoordinator:
    """Get or create the global negotiation coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = NegotiationCoordinator(working_dir=working_dir)
    return _coordinator
