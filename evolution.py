"""
Evolution: Self-improvement through prompt mutation.

This module implements:
- Prompt variant tracking
- A/B testing of prompt changes
- Automatic mutation based on success patterns
- Survival of the fittest prompts
"""

import json
import random
import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PromptVariant:
    """A variant of a prompt being tested."""
    id: str
    agent_type: str
    variant_name: str
    prompt_text: str
    success_count: int = 0
    fail_count: int = 0
    avg_tokens: float = 0.0
    avg_duration: float = 0.0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def fitness(self) -> float:
        """Calculate fitness score combining success rate and efficiency."""
        if self.success_count + self.fail_count < 3:
            return 0.5  # Not enough data

        # Success rate is most important
        success_score = self.success_rate

        # Efficiency bonus (fewer tokens = better)
        token_efficiency = 1.0 / (1 + self.avg_tokens / 10000) if self.avg_tokens > 0 else 0.5

        # Speed bonus (faster = better)
        speed_efficiency = 1.0 / (1 + self.avg_duration / 30) if self.avg_duration > 0 else 0.5

        return success_score * 0.7 + token_efficiency * 0.2 + speed_efficiency * 0.1


class PromptEvolution:
    """
    Evolves prompts through mutation and selection.

    Key principles:
    1. Track multiple variants per agent type
    2. Route traffic to variants probabilistically based on fitness
    3. Mutate successful prompts to explore improvements
    4. Retire underperforming variants
    """

    # Mutation strategies
    MUTATIONS = [
        ("add_emphasis", "Add emphasis to key instructions"),
        ("add_example", "Add a concrete example"),
        ("simplify", "Simplify and remove redundancy"),
        ("add_constraint", "Add a specific constraint or rule"),
        ("reorder", "Reorder instructions by importance"),
        ("add_negative", "Add what NOT to do"),
        ("make_specific", "Make vague instructions more specific"),
    ]

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "evolution.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize evolution database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_variants (
                id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                parent_id TEXT,
                mutation_type TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                avg_tokens REAL DEFAULT 0,
                avg_duration REAL DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_variants_agent ON prompt_variants(agent_type)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT,
                event_type TEXT,
                variant_id TEXT,
                details TEXT,
                created_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def register_base_prompt(self, agent_type: str, prompt_text: str) -> str:
        """Register a base prompt for an agent type."""
        variant_id = hashlib.md5(f"{agent_type}_base".encode()).hexdigest()[:16]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO prompt_variants
            (id, agent_type, variant_name, prompt_text, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
        """, (
            variant_id,
            agent_type,
            "base",
            prompt_text,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

        return variant_id

    def get_prompt(self, agent_type: str, exploration_rate: float = 0.1) -> tuple[str, str]:
        """
        Get a prompt for an agent, with probabilistic selection.

        Args:
            agent_type: Type of agent (grunt, orchestrator, qa, thinker)
            exploration_rate: Probability of trying a non-best variant

        Returns:
            (variant_id, prompt_text)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, prompt_text, success_count, fail_count, avg_tokens, avg_duration
            FROM prompt_variants
            WHERE agent_type = ? AND is_active = 1
        """, (agent_type,))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            # No variants registered, return None to use default
            return None, None

        # Calculate fitness for each variant
        variants = []
        for row in rows:
            variant = PromptVariant(
                id=row[0],
                agent_type=agent_type,
                variant_name="",
                prompt_text=row[1],
                success_count=row[2] or 0,
                fail_count=row[3] or 0,
                avg_tokens=row[4] or 0,
                avg_duration=row[5] or 0,
            )
            variants.append(variant)

        # Exploration: sometimes try a random variant
        if random.random() < exploration_rate and len(variants) > 1:
            selected = random.choice(variants)
            self._log_event(agent_type, "exploration", selected.id, "Random exploration")
            return selected.id, selected.prompt_text

        # Exploitation: pick the best fitness
        best = max(variants, key=lambda v: v.fitness)
        return best.id, best.prompt_text

    def record_outcome(
        self,
        variant_id: str,
        success: bool,
        tokens_used: int,
        duration_seconds: float,
    ):
        """Record the outcome of using a prompt variant."""
        if not variant_id:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current stats
        cursor.execute("""
            SELECT success_count, fail_count, avg_tokens, avg_duration
            FROM prompt_variants WHERE id = ?
        """, (variant_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return

        success_count = row[0] or 0
        fail_count = row[1] or 0
        avg_tokens = row[2] or 0
        avg_duration = row[3] or 0
        total = success_count + fail_count

        # Update running averages
        if total > 0:
            new_avg_tokens = (avg_tokens * total + tokens_used) / (total + 1)
            new_avg_duration = (avg_duration * total + duration_seconds) / (total + 1)
        else:
            new_avg_tokens = tokens_used
            new_avg_duration = duration_seconds

        # Update counts
        if success:
            cursor.execute("""
                UPDATE prompt_variants SET
                    success_count = success_count + 1,
                    avg_tokens = ?,
                    avg_duration = ?,
                    updated_at = ?
                WHERE id = ?
            """, (new_avg_tokens, new_avg_duration, datetime.now().isoformat(), variant_id))
        else:
            cursor.execute("""
                UPDATE prompt_variants SET
                    fail_count = fail_count + 1,
                    avg_tokens = ?,
                    avg_duration = ?,
                    updated_at = ?
                WHERE id = ?
            """, (new_avg_tokens, new_avg_duration, datetime.now().isoformat(), variant_id))

        conn.commit()
        conn.close()

        # Check if we should evolve
        self._maybe_evolve(variant_id)

    def _maybe_evolve(self, variant_id: str):
        """Check if conditions are right for evolution."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT agent_type, success_count, fail_count, prompt_text
            FROM prompt_variants WHERE id = ?
        """, (variant_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return

        agent_type = row[0]
        success_count = row[1] or 0
        fail_count = row[2] or 0
        prompt_text = row[3]
        total = success_count + fail_count

        conn.close()

        # Need enough data
        if total < 5:
            return

        success_rate = success_count / total

        # If doing well, maybe create a mutation
        if success_rate >= 0.7 and random.random() < 0.3:
            self._mutate_prompt(variant_id, agent_type, prompt_text, "explore_improvement")

        # If doing poorly, retire and try something new
        elif success_rate < 0.4 and total >= 10:
            self._retire_variant(variant_id)
            # Create mutation from best performing variant
            self._mutate_best(agent_type)

    def _mutate_prompt(
        self,
        parent_id: str,
        agent_type: str,
        prompt_text: str,
        reason: str,
    ) -> str | None:
        """Create a mutated variant of a prompt."""
        # Pick a random mutation strategy
        mutation_type, mutation_desc = random.choice(self.MUTATIONS)

        # Apply mutation (simple heuristics for now)
        mutated_text = self._apply_mutation(prompt_text, mutation_type)

        if mutated_text == prompt_text:
            return None  # No change

        # Create new variant
        variant_id = hashlib.md5(
            f"{agent_type}_{mutation_type}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO prompt_variants
            (id, agent_type, variant_name, prompt_text, parent_id, mutation_type, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
        """, (
            variant_id,
            agent_type,
            f"mutation_{mutation_type}",
            mutated_text,
            parent_id,
            mutation_type,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

        self._log_event(agent_type, "mutation", variant_id, f"{mutation_desc} ({reason})")

        return variant_id

    def _apply_mutation(self, prompt_text: str, mutation_type: str) -> str:
        """Apply a mutation to a prompt."""
        lines = prompt_text.split('\n')

        if mutation_type == "add_emphasis":
            # Find a key instruction and emphasize it
            for i, line in enumerate(lines):
                if line.strip().startswith(('-', '*', '1.', '2.')):
                    if "IMPORTANT" not in line and "MUST" not in line:
                        lines[i] = line.replace('. ', '. **IMPORTANT:** ', 1)
                        break

        elif mutation_type == "add_negative":
            # Add a "do NOT" instruction
            negatives = [
                "\n**DO NOT:**\n- Over-complicate simple tasks\n- Add features not requested",
                "\n**AVOID:**\n- Asking clarifying questions when you can make reasonable assumptions",
                "\n**NEVER:**\n- Leave tasks incomplete\n- Ignore error messages",
            ]
            prompt_text += random.choice(negatives)
            return prompt_text

        elif mutation_type == "simplify":
            # Remove duplicate-looking lines
            seen = set()
            new_lines = []
            for line in lines:
                normalized = line.lower().strip()
                if normalized not in seen or not normalized:
                    seen.add(normalized)
                    new_lines.append(line)
            lines = new_lines

        elif mutation_type == "add_constraint":
            constraints = [
                "\n## Constraints\n- Maximum 3 tool calls for simple tasks\n- Always verify changes work",
                "\n## Rules\n- Read before write\n- Test after change",
                "\n## Efficiency\n- Prefer single commands over multiple\n- Batch related operations",
            ]
            prompt_text += random.choice(constraints)
            return prompt_text

        elif mutation_type == "reorder":
            # Move shorter, more important-seeming sections up
            sections = []
            current_section = []
            for line in lines:
                if line.startswith('#') and current_section:
                    sections.append('\n'.join(current_section))
                    current_section = [line]
                else:
                    current_section.append(line)
            if current_section:
                sections.append('\n'.join(current_section))

            # Sort by length (shorter = more concise = often more important)
            if len(sections) > 2:
                header = sections[0]  # Keep header first
                rest = sections[1:]
                rest.sort(key=len)
                sections = [header] + rest

            return '\n\n'.join(sections)

        return '\n'.join(lines)

    def _mutate_best(self, agent_type: str):
        """Create a mutation from the best performing variant."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, prompt_text, success_count, fail_count
            FROM prompt_variants
            WHERE agent_type = ? AND is_active = 1
            ORDER BY CAST(success_count AS REAL) / MAX(success_count + fail_count, 1) DESC
            LIMIT 1
        """, (agent_type,))

        row = cursor.fetchone()
        conn.close()

        if row:
            self._mutate_prompt(row[0], agent_type, row[1], "recovery_from_poor_performer")

    def _retire_variant(self, variant_id: str):
        """Retire a poorly performing variant."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE prompt_variants SET is_active = 0, updated_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), variant_id))

        conn.commit()
        conn.close()

        self._log_event("", "retirement", variant_id, "Poor performance")

    def _log_event(self, agent_type: str, event_type: str, variant_id: str, details: str):
        """Log an evolution event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO evolution_log (agent_type, event_type, variant_id, details, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_type, event_type, variant_id, details, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Get evolution statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {"agents": {}}

        cursor.execute("""
            SELECT agent_type, COUNT(*), SUM(is_active),
                   AVG(CAST(success_count AS REAL) / MAX(success_count + fail_count, 1))
            FROM prompt_variants
            GROUP BY agent_type
        """)

        for row in cursor.fetchall():
            stats["agents"][row[0]] = {
                "total_variants": row[1],
                "active_variants": row[2],
                "avg_success_rate": row[3] or 0,
            }

        cursor.execute("SELECT COUNT(*) FROM evolution_log WHERE event_type = 'mutation'")
        stats["total_mutations"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM evolution_log WHERE event_type = 'retirement'")
        stats["total_retirements"] = cursor.fetchone()[0]

        conn.close()
        return stats

    def get_best_prompt(self, agent_type: str) -> str | None:
        """Get the best performing prompt for an agent type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT prompt_text
            FROM prompt_variants
            WHERE agent_type = ? AND is_active = 1
            ORDER BY CAST(success_count AS REAL) / MAX(success_count + fail_count, 1) DESC
            LIMIT 1
        """, (agent_type,))

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None


# Global evolution instance
_evolution: PromptEvolution | None = None


def get_evolution() -> PromptEvolution:
    """Get or create the global evolution instance."""
    global _evolution
    if _evolution is None:
        _evolution = PromptEvolution()
    return _evolution