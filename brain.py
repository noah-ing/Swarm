"""
Brain: Cognitive architecture for Swarm.

This module implements higher-order cognition:
- Reflection: Learning from outcomes
- Meta-cognition: Thinking about thinking
- Skill extraction: Creating reusable patterns
- Self-improvement: Evolving prompts based on success
- Uncertainty quantification: Knowing what we don't know
"""

import json
import re
import sqlite3
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import get_settings


@dataclass
class Reflection:
    """A reflection on a completed task."""
    task: str
    outcome: str
    success: bool
    what_worked: list[str]
    what_failed: list[str]
    insights: list[str]
    skill_candidates: list[dict]
    confidence: float
    model_used: str
    tokens_used: int
    duration_seconds: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Strategy:
    """A strategy for approaching a task."""
    name: str
    description: str
    applicable_patterns: list[str]
    steps: list[str]
    success_rate: float = 0.0
    uses: int = 0


@dataclass
class Uncertainty:
    """Uncertainty assessment for a task."""
    level: str  # low, medium, high, critical
    score: float  # 0.0 to 1.0
    reasons: list[str]
    recommended_action: str  # proceed, cautious, escalate, ask_user
    fallback_strategies: list[str]


class CognitiveArchitecture:
    """
    The brain of Swarm - handles meta-cognition and learning.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "brain.db")

        self.db_path = db_path
        self.settings = get_settings()
        self._init_db()
        self._load_strategies()

    def _init_db(self):
        """Initialize the cognitive database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Reflections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                outcome TEXT,
                success INTEGER,
                what_worked TEXT,
                what_failed TEXT,
                insights TEXT,
                skill_candidates TEXT,
                confidence REAL,
                model_used TEXT,
                tokens_used INTEGER,
                duration_seconds REAL,
                created_at TEXT
            )
        """)

        # Strategies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                applicable_patterns TEXT,
                steps TEXT,
                success_rate REAL DEFAULT 0.0,
                uses INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Prompt variants table (for A/B testing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_variants (
                id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                avg_tokens REAL,
                avg_duration REAL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Extracted skills table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                trigger_pattern TEXT NOT NULL,
                solution_template TEXT NOT NULL,
                variables TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Failure patterns table (learn from mistakes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failure_patterns (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                cause TEXT,
                solution TEXT,
                occurrences INTEGER DEFAULT 1,
                resolved_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_strategies(self):
        """Load built-in strategies."""
        self.strategies = {
            "direct_execution": Strategy(
                name="Direct Execution",
                description="Run task directly with single agent",
                applicable_patterns=["simple", "query", "single file", "command"],
                steps=["Analyze task", "Select tool", "Execute", "Verify"],
                success_rate=0.85,
            ),
            "decompose_parallel": Strategy(
                name="Parallel Decomposition",
                description="Split into independent subtasks, run in parallel",
                applicable_patterns=["multiple files", "batch", "independent operations"],
                steps=["Identify components", "Create subtasks", "Run parallel", "Aggregate"],
                success_rate=0.75,
            ),
            "decompose_sequential": Strategy(
                name="Sequential Decomposition",
                description="Split into dependent subtasks, run in order",
                applicable_patterns=["pipeline", "dependent", "build on previous"],
                steps=["Identify phases", "Order by dependency", "Execute sequentially", "Chain results"],
                success_rate=0.70,
            ),
            "explore_first": Strategy(
                name="Explore First",
                description="Understand before acting",
                applicable_patterns=["unfamiliar", "complex", "codebase", "debug"],
                steps=["Search for patterns", "Read relevant files", "Build mental model", "Plan action", "Execute"],
                success_rate=0.80,
            ),
            "trial_and_error": Strategy(
                name="Trial and Error",
                description="Try approaches, learn from failures",
                applicable_patterns=["unclear requirements", "experimental", "multiple solutions"],
                steps=["Generate hypothesis", "Test quickly", "Analyze result", "Iterate or succeed"],
                success_rate=0.65,
            ),
            "template_match": Strategy(
                name="Template Match",
                description="Find similar past solution, adapt it",
                applicable_patterns=["common task", "seen before", "standard pattern"],
                steps=["Search memory", "Find similar", "Adapt template", "Execute"],
                success_rate=0.90,
            ),
        }

    def assess_uncertainty(self, task: str, context: str = "") -> Uncertainty:
        """
        Assess uncertainty level for a task.

        Returns guidance on how to proceed based on confidence.
        """
        task_lower = task.lower()

        # Factors that increase uncertainty
        uncertainty_factors = []
        score = 0.0

        # Vague language
        vague_words = ["somehow", "maybe", "might", "something", "stuff", "thing", "whatever", "etc"]
        if any(w in task_lower for w in vague_words):
            uncertainty_factors.append("Task contains vague language")
            score += 0.2

        # Negative/unclear requirements
        if "don't know" in task_lower or "not sure" in task_lower:
            uncertainty_factors.append("Unclear requirements expressed")
            score += 0.3

        # Very short task (ambiguous)
        if len(task.split()) < 5:
            uncertainty_factors.append("Task description very brief")
            score += 0.15

        # Very long task (complex)
        if len(task.split()) > 100:
            uncertainty_factors.append("Task description very long/complex")
            score += 0.2

        # Novel domain indicators
        novel_indicators = ["never done", "first time", "new to", "unfamiliar"]
        if any(n in task_lower for n in novel_indicators):
            uncertainty_factors.append("Novel/unfamiliar domain indicated")
            score += 0.25

        # High-risk keywords
        risk_keywords = ["production", "database", "delete", "remove all", "migration", "deploy"]
        if any(k in task_lower for k in risk_keywords):
            uncertainty_factors.append("High-risk operation detected")
            score += 0.2

        # Lack of specifics
        if not re.search(r'[\w/]+\.\w+', task):  # No file paths
            if "file" in task_lower or "code" in task_lower:
                uncertainty_factors.append("References files but no specific paths")
                score += 0.15

        # Clamp score
        score = min(1.0, score)

        # Determine level and action
        if score < 0.2:
            level = "low"
            action = "proceed"
            fallbacks = []
        elif score < 0.4:
            level = "medium"
            action = "cautious"
            fallbacks = ["explore_first", "trial_and_error"]
        elif score < 0.7:
            level = "high"
            action = "escalate"
            fallbacks = ["explore_first", "decompose_sequential"]
        else:
            level = "critical"
            action = "ask_user"
            fallbacks = ["template_match", "explore_first"]

        return Uncertainty(
            level=level,
            score=score,
            reasons=uncertainty_factors if uncertainty_factors else ["Task appears straightforward"],
            recommended_action=action,
            fallback_strategies=fallbacks,
        )

    def select_strategy(self, task: str, context: str = "", uncertainty: Uncertainty | None = None) -> Strategy:
        """
        Select the best strategy for a task using meta-cognition.
        """
        task_lower = task.lower()

        # Check uncertainty first
        if uncertainty is None:
            uncertainty = self.assess_uncertainty(task, context)

        # If high uncertainty, prefer exploratory strategies
        if uncertainty.level in ("high", "critical"):
            return self.strategies["explore_first"]

        # Check for template match opportunity (fastest path)
        if self._has_similar_past_solution(task):
            return self.strategies["template_match"]

        # Analyze task characteristics
        words = task_lower.split()

        # Simple queries/commands
        simple_indicators = ["list", "show", "print", "count", "what", "where", "find", "read"]
        if any(task_lower.startswith(s) for s in simple_indicators):
            return self.strategies["direct_execution"]

        # Single file operations
        if re.search(r'fix|update|edit|change|modify', task_lower) and task_lower.count('.') <= 2:
            return self.strategies["direct_execution"]

        # Multiple independent operations
        if " and " in task_lower or ", " in task_lower:
            # Check if they're independent
            parts = re.split(r' and |, ', task_lower)
            if len(parts) >= 2 and all(len(p.split()) < 10 for p in parts):
                return self.strategies["decompose_parallel"]

        # Pipeline/sequential operations
        sequential_indicators = ["then", "after", "first", "next", "finally", "step"]
        if any(s in task_lower for s in sequential_indicators):
            return self.strategies["decompose_sequential"]

        # Complex/unfamiliar tasks
        complex_indicators = ["refactor", "architect", "design", "implement", "build", "create"]
        if any(c in task_lower for c in complex_indicators):
            if len(words) > 15:
                return self.strategies["explore_first"]
            return self.strategies["decompose_sequential"]

        # Default to direct for simple, explore for complex
        if len(words) < 10:
            return self.strategies["direct_execution"]
        return self.strategies["explore_first"]

    def _has_similar_past_solution(self, task: str) -> bool:
        """Check if we have a similar past solution."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Simple keyword matching for now
        keywords = set(task.lower().split())

        cursor.execute("""
            SELECT task FROM reflections
            WHERE success = 1
            ORDER BY created_at DESC
            LIMIT 50
        """)

        for row in cursor.fetchall():
            past_keywords = set(row[0].lower().split())
            overlap = len(keywords & past_keywords) / max(len(keywords), 1)
            if overlap > 0.5:
                conn.close()
                return True

        conn.close()
        return False

    def reflect(
        self,
        task: str,
        outcome: str,
        success: bool,
        model_used: str,
        tokens_used: int,
        duration_seconds: float,
        tool_calls: list[dict] | None = None,
        error: str | None = None,
    ) -> Reflection:
        """
        Reflect on a completed task to extract learnings.

        This is called after every task to build up knowledge.
        """
        what_worked = []
        what_failed = []
        insights = []
        skill_candidates = []

        if success:
            # Analyze what worked
            what_worked.append(f"Task completed successfully with {model_used}")

            if tokens_used < 5000:
                what_worked.append("Efficient token usage")
                insights.append("Simple approach worked well")

            if duration_seconds < 10:
                what_worked.append("Fast execution")

            # Check for skill extraction opportunity
            if tool_calls:
                # Look for patterns that could become skills
                tool_sequence = [tc.get("name") for tc in tool_calls if tc.get("name")]
                if len(tool_sequence) <= 3:
                    skill_candidates.append({
                        "name": f"skill_{hashlib.md5(task.encode()).hexdigest()[:8]}",
                        "pattern": task[:100],
                        "tools": tool_sequence,
                    })
                    insights.append(f"Potential skill: {len(tool_sequence)}-step pattern")

        else:
            what_failed.append(f"Task failed with {model_used}")

            if error:
                # Analyze error patterns
                if "timeout" in error.lower():
                    what_failed.append("Execution timed out")
                    insights.append("Consider breaking into smaller subtasks")
                elif "token" in error.lower() or "limit" in error.lower():
                    what_failed.append("Token limit exceeded")
                    insights.append("Reduce context or use summarization")
                elif "not found" in error.lower():
                    what_failed.append("Resource not found")
                    insights.append("Verify paths/resources before operations")
                else:
                    what_failed.append(f"Error: {error[:100]}")

            # Learn from failure
            if tokens_used > 50000:
                insights.append("High token usage suggests over-complicated approach")

        # Calculate confidence in this reflection
        confidence = 0.7 if success else 0.5
        if len(insights) > 0:
            confidence += 0.1
        if len(skill_candidates) > 0:
            confidence += 0.1

        reflection = Reflection(
            task=task,
            outcome=outcome,
            success=success,
            what_worked=what_worked,
            what_failed=what_failed,
            insights=insights,
            skill_candidates=skill_candidates,
            confidence=min(1.0, confidence),
            model_used=model_used,
            tokens_used=tokens_used,
            duration_seconds=duration_seconds,
        )

        # Store reflection
        self._store_reflection(reflection)

        # Extract skills if candidates found
        for candidate in skill_candidates:
            self._maybe_create_skill(candidate)

        # Update failure patterns if failed
        if not success and error:
            self._update_failure_pattern(task, error)

        return reflection

    def _store_reflection(self, reflection: Reflection):
        """Store a reflection in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        reflection_id = hashlib.md5(
            f"{reflection.task}{reflection.created_at.isoformat()}".encode()
        ).hexdigest()[:16]

        cursor.execute("""
            INSERT INTO reflections
            (id, task, outcome, success, what_worked, what_failed, insights,
             skill_candidates, confidence, model_used, tokens_used, duration_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reflection_id,
            reflection.task,
            reflection.outcome,
            1 if reflection.success else 0,
            json.dumps(reflection.what_worked),
            json.dumps(reflection.what_failed),
            json.dumps(reflection.insights),
            json.dumps(reflection.skill_candidates),
            reflection.confidence,
            reflection.model_used,
            reflection.tokens_used,
            reflection.duration_seconds,
            reflection.created_at.isoformat(),
        ))

        conn.commit()
        conn.close()

    def _maybe_create_skill(self, candidate: dict):
        """Create a skill if the pattern is seen multiple times."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        pattern = candidate.get("pattern", "")

        # Check if similar skill exists
        cursor.execute("""
            SELECT id, success_count FROM skills
            WHERE trigger_pattern LIKE ?
            LIMIT 1
        """, (f"%{pattern[:50]}%",))

        row = cursor.fetchone()
        if row:
            # Increment success count
            cursor.execute("""
                UPDATE skills SET success_count = success_count + 1, updated_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), row[0]))
        else:
            # Create new skill if pattern is specific enough
            if len(pattern) > 20:
                skill_id = candidate.get("name", hashlib.md5(pattern.encode()).hexdigest()[:16])
                cursor.execute("""
                    INSERT INTO skills
                    (id, name, trigger_pattern, solution_template, variables, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    skill_id,
                    f"Auto-skill: {pattern[:30]}",
                    pattern,
                    json.dumps(candidate.get("tools", [])),
                    json.dumps([]),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ))

        conn.commit()
        conn.close()

    def _update_failure_pattern(self, task: str, error: str):
        """Track failure patterns to avoid repeating mistakes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create a pattern signature
        error_type = "unknown"
        if "timeout" in error.lower():
            error_type = "timeout"
        elif "token" in error.lower():
            error_type = "token_limit"
        elif "not found" in error.lower():
            error_type = "not_found"
        elif "permission" in error.lower():
            error_type = "permission"
        elif "syntax" in error.lower():
            error_type = "syntax"

        pattern_id = hashlib.md5(f"{error_type}{task[:50]}".encode()).hexdigest()[:16]

        cursor.execute("""
            INSERT INTO failure_patterns (id, pattern, cause, occurrences, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                occurrences = occurrences + 1,
                updated_at = ?
        """, (
            pattern_id,
            task[:200],
            error_type,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

    def get_insights_for_task(self, task: str) -> list[str]:
        """Get relevant insights from past reflections for a task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent relevant insights
        keywords = task.lower().split()[:5]
        insights = []

        cursor.execute("""
            SELECT insights, what_worked, what_failed FROM reflections
            ORDER BY created_at DESC
            LIMIT 100
        """)

        for row in cursor.fetchall():
            try:
                row_insights = json.loads(row[0]) if row[0] else []
                what_worked = json.loads(row[1]) if row[1] else []
                what_failed = json.loads(row[2]) if row[2] else []
                insights.extend(row_insights[:2])
            except json.JSONDecodeError:
                continue

        conn.close()

        # Deduplicate and limit
        seen = set()
        unique_insights = []
        for insight in insights:
            if insight not in seen:
                seen.add(insight)
                unique_insights.append(insight)
                if len(unique_insights) >= 5:
                    break

        return unique_insights

    def get_failure_warnings(self, task: str) -> list[str]:
        """Get warnings based on past failure patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        warnings = []

        cursor.execute("""
            SELECT pattern, cause, occurrences FROM failure_patterns
            WHERE occurrences >= 2
            ORDER BY occurrences DESC
            LIMIT 10
        """)

        task_lower = task.lower()
        for row in cursor.fetchall():
            pattern = row[0].lower()
            cause = row[1]
            occurrences = row[2]

            # Check if task is similar to failure pattern
            pattern_words = set(pattern.split())
            task_words = set(task_lower.split())
            overlap = len(pattern_words & task_words) / max(len(pattern_words), 1)

            if overlap > 0.3:
                warnings.append(f"Warning: Similar tasks have failed due to {cause} ({occurrences} times)")

        conn.close()
        return warnings[:3]

    def get_recommended_model(self, task: str, default: str = "sonnet") -> str:
        """Recommend a model based on past performance with similar tasks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get success rates by model for similar tasks
        cursor.execute("""
            SELECT model_used,
                   SUM(success) as successes,
                   COUNT(*) as total,
                   AVG(tokens_used) as avg_tokens
            FROM reflections
            GROUP BY model_used
            HAVING total >= 3
        """)

        best_model = default
        best_score = 0

        for row in cursor.fetchall():
            model = row[0]
            success_rate = row[1] / row[2] if row[2] > 0 else 0
            avg_tokens = row[3] or 10000

            # Score = success_rate * efficiency
            efficiency = 1.0 / (1 + avg_tokens / 10000)
            score = success_rate * 0.7 + efficiency * 0.3

            if score > best_score:
                best_score = score
                best_model = model

        conn.close()
        return best_model

    def get_stats(self) -> dict[str, Any]:
        """Get cognitive system statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM reflections")
        stats["total_reflections"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM reflections WHERE success = 1")
        stats["successful_reflections"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM skills")
        stats["skills_learned"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM failure_patterns")
        stats["failure_patterns_tracked"] = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM reflections")
        stats["avg_confidence"] = cursor.fetchone()[0] or 0

        conn.close()
        return stats


# Global brain instance
_brain: CognitiveArchitecture | None = None


def get_brain() -> CognitiveArchitecture:
    """Get or create the global brain instance."""
    global _brain
    if _brain is None:
        _brain = CognitiveArchitecture()
    return _brain
