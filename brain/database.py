"""
Cognitive Database Component
===========================

Centralized database management for the cognitive architecture.
Handles all database operations, connection management, and CRUD operations
extracted from the monolithic CognitiveArchitecture class.
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass


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
    created_at: datetime


class CognitiveDatabase:
    """
    Centralized database management for all cognitive components.
    
    Extracted from CognitiveArchitecture to centralize all database operations:
    - Database initialization and schema management
    - Connection management
    - CRUD operations for reflections, strategies, skills, and failure patterns
    - Query builders for common patterns
    """

    def __init__(self, db_path: str | None = None):
        """Initialize database with backward-compatible path handling."""
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "brain.db")
        
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        """
        Get or create database connection.
        Uses singleton pattern for SQLite to avoid connection issues.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
        return self._connection

    def init_schema(self) -> None:
        """
        Initialize the cognitive database with exact schema from original brain.py.
        Maintains 100% backward compatibility with existing brain.db files.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Reflections table - matches original exactly
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

        # Strategies table - matches original exactly
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

        # Prompt variants table - matches original exactly
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

        # Skills table - matches original exactly
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

        # Failure patterns table - matches original exactly
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

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        Automatically commits on success, rolls back on error.
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # CRUD Operations for Reflections
    def store_reflection(self, reflection: Reflection) -> str:
        """
        Store a reflection in the database.
        Returns the generated reflection ID.
        """
        conn = self.get_connection()
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
        return reflection_id

    def get_successful_reflections(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent successful reflections for template matching."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT task FROM reflections
            WHERE success = 1
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        return [{"task": row[0]} for row in cursor.fetchall()]

    def get_recent_insights(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent insights from reflections."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT insights, what_worked, what_failed FROM reflections
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            try:
                insights = json.loads(row[0]) if row[0] else []
                what_worked = json.loads(row[1]) if row[1] else []
                what_failed = json.loads(row[2]) if row[2] else []
                results.append({
                    "insights": insights,
                    "what_worked": what_worked,
                    "what_failed": what_failed
                })
            except json.JSONDecodeError:
                continue

        return results

    def get_model_performance_stats(self) -> List[Dict[str, Any]]:
        """Get performance statistics by model for recommendation."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_used,
                   SUM(success) as successes,
                   COUNT(*) as total,
                   AVG(tokens_used) as avg_tokens
            FROM reflections
            GROUP BY model_used
            HAVING total >= 3
        """)

        return [
            {
                "model": row[0],
                "successes": row[1],
                "total": row[2],
                "avg_tokens": row[3] or 10000
            }
            for row in cursor.fetchall()
        ]

    # CRUD Operations for Skills
    def find_similar_skill(self, pattern: str, limit: int = 1) -> Optional[Dict[str, Any]]:
        """Find existing skill with similar pattern."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, success_count FROM skills
            WHERE trigger_pattern LIKE ?
            LIMIT ?
        """, (f"%{pattern[:50]}%", limit))

        row = cursor.fetchone()
        if row:
            return {"id": row[0], "success_count": row[1]}
        return None

    def update_skill_success_count(self, skill_id: str) -> None:
        """Increment success count for existing skill."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE skills SET success_count = success_count + 1, updated_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), skill_id))

        conn.commit()

    def create_skill(self, skill_id: str, name: str, pattern: str, tools: List[str]) -> None:
        """Create new skill from pattern."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO skills
            (id, name, trigger_pattern, solution_template, variables, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            skill_id,
            name,
            pattern,
            json.dumps(tools),
            json.dumps([]),
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ))

        conn.commit()

    # CRUD Operations for Failure Patterns
    def store_failure_pattern(self, task: str, error: str) -> None:
        """Track failure patterns to avoid repeating mistakes."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create a pattern signature
        error_type = self._categorize_error(error)
        pattern_id = hashlib.md5(f"{error_type}{task[:50]}".encode()).hexdigest()[:16]

        # Check if pattern already exists
        cursor.execute("SELECT occurrences FROM failure_patterns WHERE id = ?", (pattern_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing pattern
            cursor.execute("""
                UPDATE failure_patterns SET 
                occurrences = occurrences + 1,
                updated_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), pattern_id))
        else:
            # Insert new pattern
            cursor.execute("""
                INSERT INTO failure_patterns (id, pattern, cause, occurrences, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)
            """, (
                pattern_id,
                task[:200],
                error_type,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ))

        conn.commit()

    def get_failure_patterns(self, min_occurrences: int = 2, limit: int = 10) -> List[Dict[str, Any]]:
        """Get failure patterns for warnings."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT pattern, cause, occurrences FROM failure_patterns
            WHERE occurrences >= ?
            ORDER BY occurrences DESC
            LIMIT ?
        """, (min_occurrences, limit))

        return [
            {
                "pattern": row[0],
                "cause": row[1],
                "occurrences": row[2]
            }
            for row in cursor.fetchall()
        ]

    # Statistics and Analytics
    def get_system_stats(self) -> Dict[str, Any]:
        """Get cognitive system statistics."""
        conn = self.get_connection()
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

        return stats

    # Utility Methods
    def _categorize_error(self, error: str) -> str:
        """Categorize error type for pattern tracking."""
        error_lower = error.lower()
        if "timeout" in error_lower:
            return "timeout"
        elif "token" in error_lower:
            return "token_limit"
        elif "not found" in error_lower:
            return "not_found"
        elif "permission" in error_lower:
            return "permission"
        elif "syntax" in error_lower:
            return "syntax"
        else:
            return "unknown"

    def has_similar_past_solution(self, task: str) -> bool:
        """Check if we have a similar past solution for template matching."""
        keywords = set(task.lower().split())
        
        for reflection in self.get_successful_reflections():
            past_keywords = set(reflection["task"].lower().split())
            overlap = len(keywords & past_keywords) / max(len(keywords), 1)
            if overlap > 0.5:
                return True
        
        return False

    def get_insights_for_task(self, task: str) -> List[str]:
        """Get relevant insights from past reflections for a task."""
        insights = []
        
        for reflection_data in self.get_recent_insights():
            insights.extend(reflection_data["insights"][:2])
        
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

    def get_failure_warnings_for_task(self, task: str) -> List[str]:
        """Get warnings based on past failure patterns for a specific task."""
        warnings = []
        task_lower = task.lower()
        task_words = set(task_lower.split())
        
        for pattern_data in self.get_failure_patterns():
            pattern = pattern_data["pattern"].lower()
            pattern_words = set(pattern.split())
            overlap = len(pattern_words & task_words) / max(len(pattern_words), 1)
            
            if overlap > 0.3:
                warnings.append(
                    f"Warning: Similar tasks have failed due to {pattern_data['cause']} "
                    f"({pattern_data['occurrences']} times)"
                )
        
        return warnings[:3]

    def get_recommended_model(self, default: str = "sonnet") -> str:
        """Recommend a model based on past performance with similar tasks."""
        best_model = default
        best_score = 0
        
        for model_data in self.get_model_performance_stats():
            success_rate = model_data["successes"] / model_data["total"] if model_data["total"] > 0 else 0
            avg_tokens = model_data["avg_tokens"]
            
            # Score = success_rate * efficiency
            efficiency = 1.0 / (1 + avg_tokens / 10000)
            score = success_rate * 0.7 + efficiency * 0.3
            
            if score > best_score:
                best_score = score
                best_model = model_data["model"]
        
        return best_model

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None