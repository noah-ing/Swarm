"""
Continuous Background Learning: Autonomous improvement daemon.

Runs in the background to:
1. Generate learning tasks based on weaknesses
2. Practice skills that need reinforcement
3. Consolidate and compress knowledge
4. Self-improve Swarm's own code
"""

import sqlite3
import json
import time
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, Any
from enum import Enum

from brain import get_brain
from evolution import get_evolution
from memory import get_memory_store
from knowledge import get_knowledge_store


class LearningMode(Enum):
    """Types of background learning activities."""
    PRACTICE = "practice"  # Practice weak skills
    CONSOLIDATE = "consolidate"  # Merge and compress knowledge
    EXPLORE = "explore"  # Discover new patterns
    BENCHMARK = "benchmark"  # Run benchmark tests
    SELF_IMPROVE = "self_improve"  # Improve Swarm's code


@dataclass
class LearningTask:
    """A generated learning task."""
    id: str
    mode: LearningMode
    description: str
    priority: float  # 0.0-1.0
    context: str = ""
    target_skill: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LearningSession:
    """Record of a background learning session."""
    id: str
    start_time: datetime
    end_time: datetime | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    insights_gained: list[str] = field(default_factory=list)
    knowledge_consolidated: int = 0
    improvements_made: list[str] = field(default_factory=list)


# Practice tasks for skill reinforcement
PRACTICE_TEMPLATES = {
    "string_manipulation": [
        "Write a function that reverses words in a sentence",
        "Create a function to check if a string is a palindrome",
        "Write a function that compresses repeated characters",
    ],
    "list_operations": [
        "Write a function to merge two sorted lists",
        "Create a function that removes duplicates preserving order",
        "Write a function to find the intersection of two lists",
    ],
    "recursion": [
        "Write a recursive function to calculate Fibonacci numbers",
        "Create a recursive function to flatten nested lists",
        "Write a recursive function to traverse a tree structure",
    ],
    "error_handling": [
        "Write a function with robust input validation",
        "Create a retry decorator with exponential backoff",
        "Write a context manager for resource cleanup",
    ],
    "algorithms": [
        "Implement binary search with edge cases handled",
        "Write a function to find the longest common subsequence",
        "Create a function to detect cycles in a linked list",
    ],
}


class BackgroundLearner:
    """
    Autonomous learning daemon that improves Swarm over time.

    Capabilities:
    - Generate practice tasks from weaknesses
    - Consolidate fragmented knowledge
    - Run periodic benchmarks
    - Self-improve Swarm's code
    """

    # Configuration
    CONSOLIDATION_INTERVAL = timedelta(hours=1)
    BENCHMARK_INTERVAL = timedelta(hours=6)
    MIN_MEMORIES_TO_CONSOLIDATE = 10
    MAX_TASKS_PER_SESSION = 10

    def __init__(self, working_dir: str | None = None, db_path: str | None = None):
        self.working_dir = working_dir or str(Path.cwd())
        self.brain = get_brain()
        self.evolution = get_evolution()
        self.memory = get_memory_store()
        self.knowledge = get_knowledge_store()

        # Database for persistence
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "background.db")

        self.db_path = db_path
        self._init_db()

        # State
        self._running = False
        self._thread: threading.Thread | None = None
        self._current_session: LearningSession | None = None

        # Callbacks
        self.on_task_start: Callable[[LearningTask], None] | None = None
        self.on_task_complete: Callable[[LearningTask, bool], None] | None = None
        self.on_insight: Callable[[str], None] | None = None
        self.on_consolidation: Callable[[int], None] | None = None

    def _init_db(self):
        """Initialize background learning database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                tasks_completed INTEGER DEFAULT 0,
                tasks_failed INTEGER DEFAULT 0,
                insights_json TEXT,
                knowledge_consolidated INTEGER DEFAULT 0,
                improvements_json TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_practice (
                skill TEXT PRIMARY KEY,
                last_practiced TEXT,
                practice_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                memories_merged INTEGER,
                insights_extracted INTEGER,
                patterns_retired INTEGER
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_start
            ON learning_sessions(start_time DESC)
        """)

        conn.commit()
        conn.close()

    def start(self, duration_minutes: int = 60):
        """
        Start background learning daemon.

        Args:
            duration_minutes: How long to run (0 for indefinite)
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._learning_loop,
            args=(duration_minutes,),
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop background learning daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _learning_loop(self, duration_minutes: int):
        """Main learning loop."""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60) if duration_minutes > 0 else float('inf')

        # Create session
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._current_session = LearningSession(
            id=session_id,
            start_time=datetime.now()
        )

        task_count = 0

        while self._running and time.time() < end_time:
            try:
                # Generate next task
                task = self._generate_next_task()

                if task:
                    if self.on_task_start:
                        self.on_task_start(task)

                    success = self._execute_task(task)

                    if success:
                        self._current_session.tasks_completed += 1
                    else:
                        self._current_session.tasks_failed += 1

                    if self.on_task_complete:
                        self.on_task_complete(task, success)

                    task_count += 1

                    if task_count >= self.MAX_TASKS_PER_SESSION:
                        # Take a break after max tasks
                        task_count = 0
                        time.sleep(60)

                # Check if consolidation is due
                if self._should_consolidate():
                    consolidated = self._consolidate_knowledge()
                    self._current_session.knowledge_consolidated += consolidated

                    if self.on_consolidation:
                        self.on_consolidation(consolidated)

                # Brief pause between activities
                time.sleep(5)

            except Exception as e:
                # Log error but continue
                print(f"Background learning error: {e}")
                time.sleep(30)

        # End session
        self._current_session.end_time = datetime.now()
        self._save_session(self._current_session)

    def _generate_next_task(self) -> LearningTask | None:
        """Generate the next learning task based on needs."""
        # Decide what type of learning to do
        mode = self._select_learning_mode()

        if mode == LearningMode.PRACTICE:
            return self._generate_practice_task()
        elif mode == LearningMode.CONSOLIDATE:
            return None  # Handled separately
        elif mode == LearningMode.EXPLORE:
            return self._generate_exploration_task()
        elif mode == LearningMode.BENCHMARK:
            return self._generate_benchmark_task()
        elif mode == LearningMode.SELF_IMPROVE:
            return self._generate_self_improve_task()

        return None

    def _select_learning_mode(self) -> LearningMode:
        """Select the best learning mode based on current needs."""
        # Weighted random selection based on priorities
        weights = {
            LearningMode.PRACTICE: 0.4,
            LearningMode.EXPLORE: 0.2,
            LearningMode.BENCHMARK: 0.1,
            LearningMode.SELF_IMPROVE: 0.3,
        }

        # Adjust based on time since last consolidation
        if self._should_consolidate():
            return LearningMode.CONSOLIDATE

        # Random weighted selection
        modes = list(weights.keys())
        probs = list(weights.values())
        return random.choices(modes, weights=probs)[0]

    def _generate_practice_task(self) -> LearningTask:
        """Generate a practice task for weak skills."""
        # Get skill that needs practice
        skill = self._get_skill_needing_practice()

        if skill and skill in PRACTICE_TEMPLATES:
            task_desc = random.choice(PRACTICE_TEMPLATES[skill])
        else:
            # Pick random skill
            skill = random.choice(list(PRACTICE_TEMPLATES.keys()))
            task_desc = random.choice(PRACTICE_TEMPLATES[skill])

        return LearningTask(
            id=f"practice_{int(time.time())}",
            mode=LearningMode.PRACTICE,
            description=task_desc,
            priority=0.7,
            target_skill=skill,
        )

    def _generate_exploration_task(self) -> LearningTask:
        """Generate a task to explore new patterns."""
        exploration_tasks = [
            "Analyze a popular Python library and extract design patterns",
            "Find and document an interesting algorithm implementation",
            "Explore error handling patterns in production code",
            "Study logging best practices from open source projects",
            "Research testing patterns for complex systems",
        ]

        return LearningTask(
            id=f"explore_{int(time.time())}",
            mode=LearningMode.EXPLORE,
            description=random.choice(exploration_tasks),
            priority=0.5,
        )

    def _generate_benchmark_task(self) -> LearningTask:
        """Generate a benchmark task."""
        benchmark_tasks = [
            ("Create a function that reverses a string", "low"),
            ("Write a binary search implementation", "medium"),
            ("Implement a simple LRU cache", "high"),
        ]

        task, complexity = random.choice(benchmark_tasks)
        return LearningTask(
            id=f"benchmark_{int(time.time())}",
            mode=LearningMode.BENCHMARK,
            description=task,
            priority=0.3,
            context=f"Complexity: {complexity}",
        )

    def _generate_self_improve_task(self) -> LearningTask:
        """Generate a task to improve Swarm's own code."""
        swarm_files = [
            "agents/base.py",
            "agents/grunt.py",
            "brain.py",
            "evolution.py",
            "memory.py",
        ]

        improvement_types = [
            "Add type hints to untyped functions in {file}",
            "Improve error handling in {file}",
            "Optimize performance-critical code in {file}",
            "Add docstrings to undocumented functions in {file}",
        ]

        target_file = random.choice(swarm_files)
        task_template = random.choice(improvement_types)

        return LearningTask(
            id=f"improve_{int(time.time())}",
            mode=LearningMode.SELF_IMPROVE,
            description=task_template.format(file=target_file),
            priority=0.8,
            context=f"Target: {target_file}",
        )

    def _execute_task(self, task: LearningTask) -> bool:
        """Execute a learning task."""
        # Lazy import to avoid circular dependency
        from agents import Supervisor

        try:
            supervisor = Supervisor(working_dir=self.working_dir)
            result = supervisor.run(
                task=task.description,
                context=task.context,
                skip_thinking=True,  # Faster for background tasks
                allow_ask=False,  # Don't interrupt
                skip_qa=True,  # Trust the learning
            )

            # Update skill practice tracking
            if task.target_skill:
                self._update_skill_practice(task.target_skill, result.success)

            # Extract any insights
            if result.learnings:
                self._current_session.insights_gained.extend(result.learnings)
                for insight in result.learnings:
                    if self.on_insight:
                        self.on_insight(insight)

            return result.success

        except Exception as e:
            print(f"Task execution error: {e}")
            return False

    def _should_consolidate(self) -> bool:
        """Check if knowledge consolidation is due."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT MAX(timestamp) FROM consolidation_log
        """)
        row = cursor.fetchone()
        conn.close()

        if not row[0]:
            return True

        last_consolidation = datetime.fromisoformat(row[0])
        return datetime.now() - last_consolidation > self.CONSOLIDATION_INTERVAL

    def _consolidate_knowledge(self) -> int:
        """Consolidate and compress knowledge."""
        consolidated = 0

        try:
            # 1. Find and merge similar memories
            mem_stats = self.memory.get_stats()
            if mem_stats.get("total_memories", 0) >= self.MIN_MEMORIES_TO_CONSOLIDATE:
                # Extract universal insights from successful patterns
                insights = self._extract_universal_insights()
                consolidated += len(insights)

            # 2. Retire underperforming evolution variants
            evo_stats = self.evolution.get_stats()
            # Evolution handles its own retirement based on performance

            # 3. Update brain with consolidated knowledge
            brain_stats = self.brain.get_stats()
            # Brain tracks its own patterns

            # Log consolidation
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO consolidation_log
                (timestamp, memories_merged, insights_extracted, patterns_retired)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), 0, consolidated, 0))
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Consolidation error: {e}")

        return consolidated

    def _extract_universal_insights(self) -> list[str]:
        """Extract universal insights from memory patterns."""
        insights = []

        try:
            # Get successful solutions
            mem_stats = self.memory.get_stats()

            # Look for patterns in successful tasks
            # This is a simplified version - real implementation
            # would analyze actual memory contents
            if mem_stats.get("successful_memories", 0) > 5:
                insights.append("Pattern: Successful tasks tend to have clear specifications")

            # Store insights in knowledge system
            for insight in insights:
                self.knowledge.store_memory(
                    task="[Universal Insight]",
                    solution=insight,
                    success=True,
                    model="consolidation",
                    tokens_used=0,
                    cost_usd=0.0,
                    tags=["insight", "universal"],
                )

        except Exception:
            pass

        return insights

    def _get_skill_needing_practice(self) -> str | None:
        """Get the skill that most needs practice."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        # Find skill with lowest recent success or least practiced
        cursor.execute("""
            SELECT skill FROM skill_practice
            ORDER BY success_rate ASC, last_practiced ASC
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def _update_skill_practice(self, skill: str, success: bool):
        """Update skill practice tracking."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        # Get current stats
        cursor.execute(
            "SELECT practice_count, success_rate FROM skill_practice WHERE skill = ?",
            (skill,)
        )
        row = cursor.fetchone()

        if row:
            count, rate = row
            new_count = count + 1
            # Exponential moving average
            new_rate = rate * 0.9 + (1.0 if success else 0.0) * 0.1
            cursor.execute("""
                UPDATE skill_practice
                SET last_practiced = ?, practice_count = ?, success_rate = ?
                WHERE skill = ?
            """, (datetime.now().isoformat(), new_count, new_rate, skill))
        else:
            cursor.execute("""
                INSERT INTO skill_practice (skill, last_practiced, practice_count, success_rate)
                VALUES (?, ?, 1, ?)
            """, (skill, datetime.now().isoformat(), 1.0 if success else 0.0))

        conn.commit()
        conn.close()

    def _save_session(self, session: LearningSession):
        """Save learning session to database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO learning_sessions
            (id, start_time, end_time, tasks_completed, tasks_failed,
             insights_json, knowledge_consolidated, improvements_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.id,
            session.start_time.isoformat(),
            session.end_time.isoformat() if session.end_time else None,
            session.tasks_completed,
            session.tasks_failed,
            json.dumps(session.insights_gained),
            session.knowledge_consolidated,
            json.dumps(session.improvements_made),
        ))

        conn.commit()
        conn.close()

    def get_stats(self) -> dict:
        """Get background learning statistics."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM learning_sessions")
        total_sessions = cursor.fetchone()[0]

        cursor.execute(
            "SELECT SUM(tasks_completed), SUM(tasks_failed) FROM learning_sessions"
        )
        row = cursor.fetchone()
        total_completed = row[0] or 0
        total_failed = row[1] or 0

        cursor.execute("SELECT COUNT(*) FROM skill_practice")
        skills_tracked = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(success_rate) FROM skill_practice")
        avg_skill_rate = cursor.fetchone()[0] or 0

        cursor.execute("SELECT SUM(insights_extracted) FROM consolidation_log")
        total_insights = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_sessions": total_sessions,
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "success_rate": total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0,
            "skills_tracked": skills_tracked,
            "avg_skill_success_rate": avg_skill_rate,
            "insights_extracted": total_insights,
            "is_running": self._running,
        }

    def run_once(self, mode: LearningMode | None = None) -> dict:
        """
        Run a single learning iteration (for testing/manual triggers).

        Args:
            mode: Optional specific mode to use

        Returns:
            Dict with results
        """
        if mode is None:
            mode = self._select_learning_mode()

        results = {
            "mode": mode.value,
            "success": False,
            "task_description": None,
            "insights": [],
        }

        if mode == LearningMode.CONSOLIDATE:
            consolidated = self._consolidate_knowledge()
            results["success"] = True
            results["knowledge_consolidated"] = consolidated
        else:
            task = self._generate_next_task()
            if task:
                results["task_description"] = task.description
                results["success"] = self._execute_task(task)

        return results


# Global instance
_learner: BackgroundLearner | None = None


def get_background_learner(working_dir: str | None = None) -> BackgroundLearner:
    """Get or create the global background learner."""
    global _learner
    if _learner is None:
        _learner = BackgroundLearner(working_dir=working_dir)
    return _learner
