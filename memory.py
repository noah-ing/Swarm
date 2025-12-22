"""Memory system for persistent knowledge across sessions."""

import hashlib
import json
import sqlite3
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Memory:
    """A single memory entry."""

    id: str
    task: str
    solution: str
    success: bool
    model: str
    tokens_used: int
    cost_usd: float
    files_modified: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0


class MemoryStore:
    """SQLite-backed memory store with semantic search."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "memory.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                solution TEXT NOT NULL,
                success INTEGER NOT NULL,
                model TEXT,
                tokens_used INTEGER,
                cost_usd REAL,
                files_modified TEXT,
                tags TEXT,
                embedding BLOB,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_success ON memories(success)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)
        """)

        # Model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                task_type TEXT,
                success INTEGER NOT NULL,
                tokens_used INTEGER,
                cost_usd REAL,
                duration_ms INTEGER,
                created_at TEXT NOT NULL
            )
        """)

        # Skill patterns - reusable solutions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                pattern TEXT NOT NULL,
                solution_template TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                avg_tokens INTEGER,
                tags TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _hash_task(self, task: str) -> str:
        """Create a hash ID for a task."""
        return hashlib.sha256(task.encode()).hexdigest()[:16]

    def _simple_embedding(self, text: str) -> list[float]:
        """
        Create a simple bag-of-words embedding.
        For production, use a real embedding model.
        """
        # Normalize and tokenize
        words = text.lower().split()
        word_set = set(words)

        # Create feature vector based on common programming terms
        features = [
            "file", "read", "write", "create", "delete", "update", "fix",
            "bug", "error", "test", "function", "class", "import", "api",
            "database", "sql", "json", "http", "request", "response",
            "loop", "condition", "variable", "string", "number", "list",
            "dict", "array", "object", "return", "print", "log", "debug",
            "config", "setting", "env", "path", "directory", "git", "commit",
            "python", "javascript", "typescript", "react", "node", "bash",
            "install", "package", "dependency", "build", "deploy", "run",
            "refactor", "optimize", "performance", "memory", "cache",
            "auth", "user", "password", "token", "session", "cookie",
            "frontend", "backend", "server", "client", "component", "hook",
        ]

        # Binary presence vector + word frequency features
        embedding = []
        for feature in features:
            embedding.append(1.0 if feature in word_set else 0.0)

        # Add some statistical features
        embedding.append(min(1.0, len(words) / 100))  # Normalized length
        embedding.append(min(1.0, len(word_set) / 50))  # Vocabulary diversity

        return embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def store(self, memory: Memory) -> str:
        """Store a memory entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        memory_id = memory.id or self._hash_task(memory.task)
        embedding = memory.embedding or self._simple_embedding(memory.task + " " + memory.solution)

        cursor.execute("""
            INSERT OR REPLACE INTO memories
            (id, task, solution, success, model, tokens_used, cost_usd,
             files_modified, tags, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            memory.task,
            memory.solution,
            1 if memory.success else 0,
            memory.model,
            memory.tokens_used,
            memory.cost_usd,
            json.dumps(memory.files_modified),
            json.dumps(memory.tags),
            json.dumps(embedding),
            memory.created_at.isoformat(),
        ))

        conn.commit()
        conn.close()

        return memory_id

    def search(
        self,
        query: str,
        limit: int = 5,
        success_only: bool = True,
        min_similarity: float = 0.3,
    ) -> list[Memory]:
        """Search for relevant memories using semantic similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query_embedding = self._simple_embedding(query)

        where_clause = "WHERE success = 1" if success_only else ""
        cursor.execute(f"""
            SELECT id, task, solution, success, model, tokens_used, cost_usd,
                   files_modified, tags, embedding, created_at
            FROM memories
            {where_clause}
            ORDER BY created_at DESC
            LIMIT 100
        """)

        results = []
        for row in cursor.fetchall():
            try:
                stored_embedding = json.loads(row[9]) if row[9] else []
                similarity = self._cosine_similarity(query_embedding, stored_embedding)

                if similarity >= min_similarity:
                    memory = Memory(
                        id=row[0],
                        task=row[1],
                        solution=row[2],
                        success=bool(row[3]),
                        model=row[4],
                        tokens_used=row[5] or 0,
                        cost_usd=row[6] or 0.0,
                        files_modified=json.loads(row[7]) if row[7] else [],
                        tags=json.loads(row[8]) if row[8] else [],
                        embedding=stored_embedding,
                        created_at=datetime.fromisoformat(row[10]),
                        relevance_score=similarity,
                    )
                    results.append(memory)
            except (json.JSONDecodeError, TypeError):
                continue

        conn.close()

        # Sort by similarity and return top results
        results.sort(key=lambda m: m.relevance_score, reverse=True)
        return results[:limit]

    def get_similar_solutions(self, task: str, limit: int = 3) -> str:
        """Get formatted similar solutions for context injection."""
        memories = self.search(task, limit=limit, success_only=True)

        if not memories:
            return ""

        context_parts = ["## Similar Past Solutions\n"]

        for i, mem in enumerate(memories, 1):
            context_parts.append(f"### Example {i} (similarity: {mem.relevance_score:.0%})")
            context_parts.append(f"**Task:** {mem.task[:200]}...")
            context_parts.append(f"**Solution:** {mem.solution[:500]}...")
            if mem.files_modified:
                context_parts.append(f"**Files:** {', '.join(mem.files_modified[:5])}")
            context_parts.append("")

        return "\n".join(context_parts)

    def track_model_performance(
        self,
        model: str,
        task_type: str,
        success: bool,
        tokens_used: int,
        cost_usd: float,
        duration_ms: int,
    ):
        """Track model performance for adaptive routing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO model_performance
            (model, task_type, success, tokens_used, cost_usd, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            model,
            task_type,
            1 if success else 0,
            tokens_used,
            cost_usd,
            duration_ms,
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

    def get_model_stats(self, model: str | None = None) -> dict[str, Any]:
        """Get performance statistics for models."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        where_clause = f"WHERE model = '{model}'" if model else ""

        cursor.execute(f"""
            SELECT
                model,
                COUNT(*) as total,
                SUM(success) as successes,
                AVG(tokens_used) as avg_tokens,
                SUM(cost_usd) as total_cost,
                AVG(duration_ms) as avg_duration
            FROM model_performance
            {where_clause}
            GROUP BY model
        """)

        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = {
                "total_tasks": row[1],
                "successes": row[2],
                "success_rate": row[2] / row[1] if row[1] > 0 else 0,
                "avg_tokens": row[3],
                "total_cost": row[4],
                "avg_duration_ms": row[5],
            }

        conn.close()
        return stats

    def get_best_model_for_task(self, task_type: str) -> str | None:
        """Get the best performing model for a task type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                model,
                COUNT(*) as total,
                SUM(success) * 1.0 / COUNT(*) as success_rate,
                AVG(cost_usd) as avg_cost
            FROM model_performance
            WHERE task_type = ?
            GROUP BY model
            HAVING total >= 3
            ORDER BY success_rate DESC, avg_cost ASC
            LIMIT 1
        """, (task_type,))

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def store_skill(
        self,
        name: str,
        pattern: str,
        solution_template: str,
        tags: list[str] | None = None,
    ) -> str:
        """Store a reusable skill pattern."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        skill_id = self._hash_task(pattern)
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO skills
            (id, name, pattern, solution_template, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            skill_id,
            name,
            pattern,
            solution_template,
            json.dumps(tags or []),
            now,
            now,
        ))

        conn.commit()
        conn.close()

        return skill_id

    def find_skill(self, task: str) -> dict | None:
        """Find a matching skill for a task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, pattern, solution_template, success_count, fail_count
            FROM skills
            ORDER BY success_count DESC
        """)

        task_lower = task.lower()
        task_words = set(task_lower.split())

        best_match = None
        best_score = 0

        for row in cursor.fetchall():
            pattern_words = set(row[2].lower().split())
            overlap = len(task_words & pattern_words)
            score = overlap / max(len(pattern_words), 1)

            if score > best_score and score > 0.5:
                best_score = score
                best_match = {
                    "id": row[0],
                    "name": row[1],
                    "pattern": row[2],
                    "solution_template": row[3],
                    "success_count": row[4],
                    "fail_count": row[5],
                    "match_score": score,
                }

        conn.close()
        return best_match

    def update_skill_stats(self, skill_id: str, success: bool):
        """Update skill success/fail counts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        field = "success_count" if success else "fail_count"
        cursor.execute(f"""
            UPDATE skills
            SET {field} = {field} + 1, updated_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), skill_id))

        conn.commit()
        conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Get overall memory statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM memories WHERE success = 1")
        successful = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(cost_usd) FROM memories")
        total_cost = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM skills")
        total_skills = cursor.fetchone()[0]

        conn.close()

        return {
            "total_memories": total_memories,
            "successful_memories": successful,
            "success_rate": successful / total_memories if total_memories > 0 else 0,
            "total_cost_usd": total_cost,
            "total_skills": total_skills,
        }


# Global memory store instance
_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Get or create the global memory store."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
