"""
Meta-Learner: Optimizes Swarm's own learning process.

The meta-learner sits above all cognitive systems and:
1. Monitors learning effectiveness across all systems
2. Identifies which strategies produce best outcomes
3. Tunes hyperparameters automatically
4. Proposes and runs experiments to improve learning
5. Tracks meta-learning progress over time

This is learning-to-learn: optimizing the optimization process.
"""

import sqlite3
import json
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, Any
from enum import Enum

from brain import get_brain
from evolution import get_evolution
from memory import get_memory_store
from knowledge import get_knowledge_store


class ExperimentType(Enum):
    """Types of meta-learning experiments."""
    PROMPT_VARIANT = "prompt_variant"  # Test new prompt formulations
    STRATEGY_WEIGHT = "strategy_weight"  # Adjust strategy selection weights
    MODEL_ROUTING = "model_routing"  # Optimize model selection
    CONSOLIDATION = "consolidation"  # Test consolidation strategies
    DECOMPOSITION = "decomposition"  # Test task decomposition approaches


@dataclass
class Hypothesis:
    """A hypothesis about improving learning."""
    id: str
    description: str
    experiment_type: ExperimentType
    parameter: str  # What to change
    current_value: Any
    proposed_value: Any
    expected_improvement: float  # Expected % improvement
    confidence: float  # 0.0-1.0
    rationale: str


@dataclass
class Experiment:
    """A meta-learning experiment."""
    id: str
    hypothesis: Hypothesis
    start_time: datetime
    end_time: datetime | None = None
    control_metrics: dict = field(default_factory=dict)
    treatment_metrics: dict = field(default_factory=dict)
    result: str | None = None  # "improved", "degraded", "neutral"
    improvement_pct: float = 0.0


@dataclass
class LearningEfficiency:
    """Metrics for learning efficiency."""
    success_rate: float
    tokens_per_success: float
    time_per_success: float
    knowledge_retention: float  # % of insights still useful
    skill_improvement: float  # Rate of skill improvement
    cross_project_transfer: float  # % of successful transfers


class MetaLearner:
    """
    Optimizes Swarm's learning process through experimentation.

    Monitors all cognitive systems, identifies improvement opportunities,
    runs controlled experiments, and applies successful changes.
    """

    # Configuration (these are what we optimize)
    CONFIG = {
        "evolution": {
            "mutation_rate": 0.1,  # Probability of mutation
            "retirement_threshold": 0.3,  # Min success rate to survive
            "exploration_bonus": 0.2,  # Bonus for trying new variants
        },
        "brain": {
            "reflection_depth": "medium",  # shallow/medium/deep
            "insight_threshold": 0.7,  # Min confidence for insights
            "skill_extraction_threshold": 0.8,  # Min success for skill creation
        },
        "strategy": {
            "direct_weight": 0.3,
            "decompose_weight": 0.3,
            "explore_weight": 0.2,
            "hierarchical_weight": 0.2,
        },
        "model_routing": {
            "escalation_threshold": 0.5,  # Failure rate to escalate
            "cheap_mode_threshold": 0.8,  # Task simplicity for cheap mode
        },
    }

    def __init__(self, db_path: str | None = None):
        self.brain = get_brain()
        self.evolution = get_evolution()
        self.memory = get_memory_store()
        self.knowledge = get_knowledge_store()

        # Database for persistence
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "metalearner.db")

        self.db_path = db_path
        self._init_db()

        # Load saved config
        self._load_config()

        # Callbacks
        self.on_hypothesis: Callable[[Hypothesis], None] | None = None
        self.on_experiment_start: Callable[[Experiment], None] | None = None
        self.on_experiment_end: Callable[[Experiment], None] | None = None

    def _init_db(self):
        """Initialize meta-learner database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_json TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                reason TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                hypothesis_json TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                control_metrics_json TEXT,
                treatment_metrics_json TEXT,
                result TEXT,
                improvement_pct REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS efficiency_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                success_rate REAL,
                tokens_per_success REAL,
                time_per_success REAL,
                knowledge_retention REAL,
                skill_improvement REAL,
                cross_project_transfer REAL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_time
            ON experiments(start_time DESC)
        """)

        conn.commit()
        conn.close()

    def _load_config(self):
        """Load the most recent config from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT config_json FROM config_history
            ORDER BY timestamp DESC LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        if row:
            saved_config = json.loads(row[0])
            # Merge with defaults (in case new params added)
            for category, params in saved_config.items():
                if category in self.CONFIG:
                    self.CONFIG[category].update(params)

    def _save_config(self, reason: str = ""):
        """Save current config to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO config_history (config_json, timestamp, reason)
            VALUES (?, ?, ?)
        """, (json.dumps(self.CONFIG), datetime.now().isoformat(), reason))

        conn.commit()
        conn.close()

    def analyze_efficiency(self) -> LearningEfficiency:
        """Analyze current learning efficiency across all systems."""
        brain_stats = self.brain.get_stats()
        evo_stats = self.evolution.get_stats()
        mem_stats = self.memory.get_stats()
        know_stats = self.knowledge.get_stats()

        # Calculate success rate
        total_tasks = brain_stats.get("total_reflections", 0)
        successful_tasks = brain_stats.get("successful_reflections", 0)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0

        # Estimate tokens per success (rough approximation)
        # Would need actual token tracking for precision
        tokens_per_success = 5000  # Placeholder

        # Time per success (from brain stats if available)
        time_per_success = 30.0  # Placeholder

        # Knowledge retention (% of insights still being used)
        total_insights = know_stats.get("universal_insights", 0)
        knowledge_retention = 0.8 if total_insights > 0 else 0  # Placeholder

        # Skill improvement rate
        skill_improvement = brain_stats.get("avg_confidence", 0)

        # Cross-project transfer success
        transfers = know_stats.get("total_transfers", 0)
        transfer_rate = know_stats.get("transfer_success_rate", 0)

        efficiency = LearningEfficiency(
            success_rate=success_rate,
            tokens_per_success=tokens_per_success,
            time_per_success=time_per_success,
            knowledge_retention=knowledge_retention,
            skill_improvement=skill_improvement,
            cross_project_transfer=transfer_rate,
        )

        # Store snapshot
        self._store_efficiency_snapshot(efficiency)

        return efficiency

    def _store_efficiency_snapshot(self, efficiency: LearningEfficiency):
        """Store efficiency snapshot for trend analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO efficiency_snapshots
            (timestamp, success_rate, tokens_per_success, time_per_success,
             knowledge_retention, skill_improvement, cross_project_transfer)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            efficiency.success_rate,
            efficiency.tokens_per_success,
            efficiency.time_per_success,
            efficiency.knowledge_retention,
            efficiency.skill_improvement,
            efficiency.cross_project_transfer,
        ))

        conn.commit()
        conn.close()

    def generate_hypotheses(self) -> list[Hypothesis]:
        """Generate hypotheses for improving learning based on current data."""
        hypotheses = []
        efficiency = self.analyze_efficiency()

        # Hypothesis 1: If success rate is low, try different strategies
        if efficiency.success_rate < 0.7:
            hypotheses.append(Hypothesis(
                id=f"hyp_{int(time.time())}_1",
                description="Increase decomposition for complex tasks",
                experiment_type=ExperimentType.STRATEGY_WEIGHT,
                parameter="strategy.decompose_weight",
                current_value=self.CONFIG["strategy"]["decompose_weight"],
                proposed_value=min(0.5, self.CONFIG["strategy"]["decompose_weight"] + 0.1),
                expected_improvement=0.1,
                confidence=0.6,
                rationale=f"Success rate is {efficiency.success_rate:.0%}, decomposing more may help",
            ))

        # Hypothesis 2: If using too many tokens, try more aggressive model routing
        if efficiency.tokens_per_success > 10000:
            hypotheses.append(Hypothesis(
                id=f"hyp_{int(time.time())}_2",
                description="Use cheaper models for simple tasks more aggressively",
                experiment_type=ExperimentType.MODEL_ROUTING,
                parameter="model_routing.cheap_mode_threshold",
                current_value=self.CONFIG["model_routing"]["cheap_mode_threshold"],
                proposed_value=min(0.9, self.CONFIG["model_routing"]["cheap_mode_threshold"] + 0.05),
                expected_improvement=0.15,
                confidence=0.5,
                rationale="High token usage suggests over-powered models for simple tasks",
            ))

        # Hypothesis 3: If knowledge retention is low, extract more skills
        if efficiency.knowledge_retention < 0.6:
            hypotheses.append(Hypothesis(
                id=f"hyp_{int(time.time())}_3",
                description="Lower skill extraction threshold to capture more patterns",
                experiment_type=ExperimentType.CONSOLIDATION,
                parameter="brain.skill_extraction_threshold",
                current_value=self.CONFIG["brain"]["skill_extraction_threshold"],
                proposed_value=max(0.6, self.CONFIG["brain"]["skill_extraction_threshold"] - 0.1),
                expected_improvement=0.2,
                confidence=0.4,
                rationale=f"Knowledge retention at {efficiency.knowledge_retention:.0%}, need more skill extraction",
            ))

        # Hypothesis 4: If evolution isn't producing winners, adjust mutation rate
        evo_stats = self.evolution.get_stats()
        if evo_stats.get("total_mutations", 0) > 5:
            agents = evo_stats.get("agents", {})
            avg_success = sum(a.get("avg_success_rate", 0) for a in agents.values()) / len(agents) if agents else 0

            if avg_success < 0.6:
                hypotheses.append(Hypothesis(
                    id=f"hyp_{int(time.time())}_4",
                    description="Increase mutation rate for more prompt exploration",
                    experiment_type=ExperimentType.PROMPT_VARIANT,
                    parameter="evolution.mutation_rate",
                    current_value=self.CONFIG["evolution"]["mutation_rate"],
                    proposed_value=min(0.3, self.CONFIG["evolution"]["mutation_rate"] + 0.05),
                    expected_improvement=0.1,
                    confidence=0.5,
                    rationale=f"Evolution success at {avg_success:.0%}, need more exploration",
                ))

        # Hypothesis 5: Cross-project transfer improvements
        if efficiency.cross_project_transfer < 0.5:
            hypotheses.append(Hypothesis(
                id=f"hyp_{int(time.time())}_5",
                description="Deeper reflection for better insight extraction",
                experiment_type=ExperimentType.CONSOLIDATION,
                parameter="brain.reflection_depth",
                current_value=self.CONFIG["brain"]["reflection_depth"],
                proposed_value="deep",
                expected_improvement=0.15,
                confidence=0.4,
                rationale=f"Cross-project transfer at {efficiency.cross_project_transfer:.0%}, need richer insights",
            ))

        return hypotheses

    def run_experiment(self, hypothesis: Hypothesis) -> Experiment:
        """
        Run an A/B experiment to test a hypothesis.

        Note: In practice, this would run tasks with control/treatment configs.
        Here we implement the framework for future execution.
        """
        experiment = Experiment(
            id=f"exp_{int(time.time())}",
            hypothesis=hypothesis,
            start_time=datetime.now(),
        )

        if self.on_experiment_start:
            self.on_experiment_start(experiment)

        # Store initial metrics as control
        experiment.control_metrics = self._get_current_metrics()

        # Apply the change
        self._apply_config_change(hypothesis.parameter, hypothesis.proposed_value)

        # In a real implementation, we'd run tasks here and measure
        # For now, we just record that the change was made

        experiment.treatment_metrics = self._get_current_metrics()
        experiment.end_time = datetime.now()

        # Evaluate result (simplified - real version would compare actual performance)
        # For now, just mark as "applied"
        experiment.result = "applied"
        experiment.improvement_pct = 0.0  # Would be calculated from actual metrics

        # Store experiment
        self._store_experiment(experiment)

        if self.on_experiment_end:
            self.on_experiment_end(experiment)

        return experiment

    def _get_current_metrics(self) -> dict:
        """Get current metrics snapshot."""
        efficiency = self.analyze_efficiency()
        return {
            "success_rate": efficiency.success_rate,
            "tokens_per_success": efficiency.tokens_per_success,
            "skill_improvement": efficiency.skill_improvement,
            "cross_project_transfer": efficiency.cross_project_transfer,
        }

    def _apply_config_change(self, parameter: str, value: Any):
        """Apply a configuration change."""
        parts = parameter.split(".")
        if len(parts) == 2:
            category, param = parts
            if category in self.CONFIG and param in self.CONFIG[category]:
                old_value = self.CONFIG[category][param]
                self.CONFIG[category][param] = value
                self._save_config(f"Changed {parameter} from {old_value} to {value}")

    def _store_experiment(self, experiment: Experiment):
        """Store experiment results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO experiments
            (id, hypothesis_json, start_time, end_time,
             control_metrics_json, treatment_metrics_json, result, improvement_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment.id,
            json.dumps({
                "id": experiment.hypothesis.id,
                "description": experiment.hypothesis.description,
                "parameter": experiment.hypothesis.parameter,
                "current_value": str(experiment.hypothesis.current_value),
                "proposed_value": str(experiment.hypothesis.proposed_value),
            }),
            experiment.start_time.isoformat(),
            experiment.end_time.isoformat() if experiment.end_time else None,
            json.dumps(experiment.control_metrics),
            json.dumps(experiment.treatment_metrics),
            experiment.result,
            experiment.improvement_pct,
        ))

        conn.commit()
        conn.close()

    def optimize(self, max_experiments: int = 3) -> list[Experiment]:
        """
        Run optimization cycle: generate hypotheses, run experiments, apply winners.
        """
        results = []

        # Generate hypotheses
        hypotheses = self.generate_hypotheses()

        if not hypotheses:
            return results

        # Sort by expected improvement * confidence
        hypotheses.sort(
            key=lambda h: h.expected_improvement * h.confidence,
            reverse=True
        )

        # Run top experiments
        for hypothesis in hypotheses[:max_experiments]:
            if self.on_hypothesis:
                self.on_hypothesis(hypothesis)

            experiment = self.run_experiment(hypothesis)
            results.append(experiment)

        return results

    def get_efficiency_trend(self, days: int = 7) -> list[dict]:
        """Get efficiency trend over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT timestamp, success_rate, tokens_per_success, skill_improvement
            FROM efficiency_snapshots
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        """, (cutoff,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "timestamp": row[0],
                "success_rate": row[1],
                "tokens_per_success": row[2],
                "skill_improvement": row[3],
            }
            for row in rows
        ]

    def get_stats(self) -> dict:
        """Get meta-learner statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM experiments")
        total_experiments = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM experiments WHERE result = 'improved'")
        successful_experiments = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM config_history")
        config_changes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM efficiency_snapshots")
        snapshots = cursor.fetchone()[0]

        conn.close()

        return {
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "experiment_success_rate": successful_experiments / total_experiments if total_experiments > 0 else 0,
            "config_changes": config_changes,
            "efficiency_snapshots": snapshots,
            "current_config": self.CONFIG,
        }

    def format_efficiency(self, efficiency: LearningEfficiency) -> str:
        """Format efficiency metrics for display."""
        return f"""## Learning Efficiency

Success Rate: {efficiency.success_rate:.1%}
Tokens/Success: {efficiency.tokens_per_success:,.0f}
Time/Success: {efficiency.time_per_success:.1f}s
Knowledge Retention: {efficiency.knowledge_retention:.1%}
Skill Improvement: {efficiency.skill_improvement:.1%}
Cross-Project Transfer: {efficiency.cross_project_transfer:.1%}
"""

    def format_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Format hypothesis for display."""
        return f"""## Hypothesis: {hypothesis.description}

Type: {hypothesis.experiment_type.value}
Parameter: {hypothesis.parameter}
Current: {hypothesis.current_value}
Proposed: {hypothesis.proposed_value}
Expected Improvement: {hypothesis.expected_improvement:.0%}
Confidence: {hypothesis.confidence:.0%}

Rationale: {hypothesis.rationale}
"""


# Global instance
_metalearner: MetaLearner | None = None


def get_metalearner() -> MetaLearner:
    """Get or create the global meta-learner."""
    global _metalearner
    if _metalearner is None:
        _metalearner = MetaLearner()
    return _metalearner
