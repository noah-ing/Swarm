"""
Cross-Project Knowledge Transfer System.

Enables Swarm to share learnings between different projects:
- Project-aware memory storage
- Cross-project semantic search with relevance weighting
- Universal insight extraction (patterns that work everywhere)
- Knowledge transfer tracking
"""

import hashlib
import json
import sqlite3
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from embeddings import get_embedding_service


@dataclass
class ProjectInfo:
    """Information about a project."""
    id: str
    name: str
    path: str
    language: str = ""
    framework: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class UniversalInsight:
    """A pattern or insight that transfers across projects."""
    id: str
    pattern: str  # What triggers this insight
    insight: str  # The actual learning
    source_projects: list[str]  # Projects where this was learned
    success_count: int = 0
    apply_count: int = 0
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def effectiveness(self) -> float:
        """How effective is this insight when applied?"""
        if self.apply_count == 0:
            return 0.5  # Unknown
        return self.success_count / self.apply_count


@dataclass
class CrossProjectMatch:
    """A memory match from another project."""
    memory_id: str
    task: str
    solution: str
    source_project: str
    source_project_name: str
    similarity: float
    relevance_score: float  # Adjusted for cross-project transfer


class KnowledgeStore:
    """
    Cross-project knowledge management.

    Stores project-aware memories and extracts universal insights
    that can be applied across different codebases.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "knowledge.db")

        self.db_path = db_path
        self._init_db()
        self._current_project: Optional[str] = None

    def _init_db(self):
        """Initialize the knowledge database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                language TEXT,
                framework TEXT,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL
            )
        """)

        # Project memories (extends base memories with project context)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_memories (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                task TEXT NOT NULL,
                solution TEXT NOT NULL,
                success INTEGER NOT NULL,
                model TEXT,
                tokens_used INTEGER,
                cost_usd REAL,
                files_modified TEXT,
                tags TEXT,
                embedding TEXT,
                is_transferable INTEGER DEFAULT 0,
                transfer_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pm_project ON project_memories(project_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pm_transferable ON project_memories(is_transferable)
        """)

        # Universal insights (cross-project patterns)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universal_insights (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                insight TEXT NOT NULL,
                category TEXT,
                source_projects TEXT,
                success_count INTEGER DEFAULT 0,
                apply_count INTEGER DEFAULT 0,
                embedding TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Knowledge transfer log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transfer_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_project TEXT NOT NULL,
                target_project TEXT NOT NULL,
                memory_id TEXT,
                insight_id TEXT,
                success INTEGER,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _hash(self, text: str) -> str:
        """Create a hash ID."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _get_project_id(self, path: str) -> str:
        """Generate a consistent project ID from path."""
        # Use the resolved absolute path for consistency
        resolved = str(Path(path).resolve())
        return self._hash(resolved)

    def set_current_project(self, path: str, name: Optional[str] = None) -> ProjectInfo:
        """
        Set the current working project.

        Args:
            path: Path to the project root
            name: Optional project name (defaults to directory name)

        Returns:
            ProjectInfo for the project
        """
        resolved_path = str(Path(path).resolve())
        project_id = self._get_project_id(resolved_path)

        if name is None:
            name = Path(resolved_path).name

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        # Check if project exists
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()

        if row:
            # Update last accessed
            cursor.execute("""
                UPDATE projects SET last_accessed = ? WHERE id = ?
            """, (now, project_id))
            project = ProjectInfo(
                id=row[0],
                name=row[1],
                path=row[2],
                language=row[3] or "",
                framework=row[4] or "",
                created_at=datetime.fromisoformat(row[5]),
                last_accessed=datetime.now(),
            )
        else:
            # Create new project
            cursor.execute("""
                INSERT INTO projects (id, name, path, language, framework, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (project_id, name, resolved_path, "", "", now, now))
            project = ProjectInfo(
                id=project_id,
                name=name,
                path=resolved_path,
            )

        conn.commit()
        conn.close()

        self._current_project = project_id
        return project

    def store_memory(
        self,
        task: str,
        solution: str,
        success: bool,
        model: str = "",
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        files_modified: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """
        Store a memory with project context.

        Args:
            task: The task that was performed
            solution: How it was solved
            success: Whether it succeeded
            model: Model used
            tokens_used: Tokens consumed
            cost_usd: Cost in USD
            files_modified: Files that were changed
            tags: Optional tags
            project_id: Project ID (uses current if not specified)

        Returns:
            Memory ID
        """
        if project_id is None:
            project_id = self._current_project or "global"

        memory_id = self._hash(f"{project_id}{task}{datetime.now().isoformat()}")

        # Get embedding
        service = get_embedding_service()
        embedding = service.embed(f"{task} {solution}").embedding

        # Determine if this memory is transferable
        is_transferable = self._assess_transferability(task, solution, files_modified or [])

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO project_memories
            (id, project_id, task, solution, success, model, tokens_used, cost_usd,
             files_modified, tags, embedding, is_transferable, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            project_id,
            task,
            solution,
            1 if success else 0,
            model,
            tokens_used,
            cost_usd,
            json.dumps(files_modified or []),
            json.dumps(tags or []),
            json.dumps(embedding),
            1 if is_transferable else 0,
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

        # Check if this creates a universal insight
        if success and is_transferable:
            self._maybe_create_insight(task, solution, project_id)

        return memory_id

    def _assess_transferability(
        self,
        task: str,
        solution: str,
        files_modified: list[str],
    ) -> bool:
        """
        Assess if a memory is likely transferable to other projects.

        Transferable:
        - Generic programming patterns
        - Algorithm implementations
        - Best practices

        Not transferable:
        - Project-specific file paths
        - Configuration specific to one project
        - References to specific architecture
        """
        task_lower = task.lower()
        solution_lower = solution.lower()

        # Indicators of transferability
        transferable_keywords = [
            "function", "class", "algorithm", "pattern", "decorator",
            "implement", "create", "write", "build", "design",
            "sort", "search", "filter", "map", "reduce",
            "cache", "memoize", "optimize", "refactor",
            "test", "validate", "parse", "serialize",
            "api", "http", "request", "response",
            "error handling", "exception", "logging",
        ]

        # Indicators of project-specificity
        specific_keywords = [
            "config", "settings", "env",
            "our", "this project", "specific",
            "migration", "deploy", "production",
        ]

        # Count indicators
        transferable_score = sum(1 for kw in transferable_keywords if kw in task_lower or kw in solution_lower)
        specific_score = sum(1 for kw in specific_keywords if kw in task_lower or kw in solution_lower)

        # Check file paths - lots of project-specific paths suggest non-transferable
        if len(files_modified) > 5:
            specific_score += 2

        return transferable_score > specific_score

    def _maybe_create_insight(self, task: str, solution: str, project_id: str):
        """Check if this memory creates or reinforces a universal insight."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        # Get embedding for matching
        service = get_embedding_service()
        task_embedding = service.embed(task).embedding

        # Look for similar existing insights
        cursor.execute("SELECT id, pattern, insight, embedding, source_projects FROM universal_insights")
        rows = cursor.fetchall()

        best_match = None
        best_similarity = 0.0

        for row in rows:
            stored_embedding = json.loads(row[3]) if row[3] else []
            similarity = service.similarity(task_embedding, stored_embedding)
            if similarity > best_similarity and similarity > 0.7:
                best_match = row
                best_similarity = similarity

        if best_match:
            # Reinforce existing insight
            insight_id = best_match[0]
            source_projects = json.loads(best_match[4]) if best_match[4] else []
            if project_id not in source_projects:
                source_projects.append(project_id)

            cursor.execute("""
                UPDATE universal_insights
                SET success_count = success_count + 1,
                    source_projects = ?,
                    updated_at = ?
                WHERE id = ?
            """, (json.dumps(source_projects), datetime.now().isoformat(), insight_id))
        else:
            # Create new insight if task is generic enough
            if self._is_generic_pattern(task):
                insight_id = self._hash(task)
                pattern = self._extract_pattern(task)
                insight = self._extract_insight(task, solution)

                cursor.execute("""
                    INSERT OR IGNORE INTO universal_insights
                    (id, pattern, insight, category, source_projects, embedding, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight_id,
                    pattern,
                    insight,
                    self._categorize_task(task),
                    json.dumps([project_id]),
                    json.dumps(task_embedding),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ))

        conn.commit()
        conn.close()

    def _is_generic_pattern(self, task: str) -> bool:
        """Check if a task represents a generic pattern."""
        generic_indicators = [
            "create a", "write a", "implement",
            "function", "class", "method",
            "that", "which", "to",
        ]
        task_lower = task.lower()
        return sum(1 for ind in generic_indicators if ind in task_lower) >= 2

    def _extract_pattern(self, task: str) -> str:
        """Extract the general pattern from a specific task."""
        # Remove project-specific details
        pattern = task
        # Could be enhanced with NLP
        return pattern[:200]

    def _extract_insight(self, task: str, solution: str) -> str:
        """Extract the key insight from a solution."""
        # Take the approach summary
        lines = solution.split('\n')
        for line in lines:
            if line.strip().startswith(('The', 'I ', 'Use', 'Create', 'Implement')):
                return line.strip()[:300]
        return solution[:300]

    def _categorize_task(self, task: str) -> str:
        """Categorize a task into a general category."""
        task_lower = task.lower()

        categories = {
            "data_structures": ["list", "array", "tree", "graph", "stack", "queue", "hash"],
            "algorithms": ["sort", "search", "find", "optimize", "calculate"],
            "patterns": ["pattern", "singleton", "factory", "decorator", "observer"],
            "api": ["api", "http", "rest", "request", "endpoint"],
            "testing": ["test", "unittest", "pytest", "mock", "assert"],
            "error_handling": ["error", "exception", "try", "catch", "handle"],
            "performance": ["cache", "memoize", "optimize", "performance", "fast"],
        }

        for category, keywords in categories.items():
            if any(kw in task_lower for kw in keywords):
                return category

        return "general"

    def search_cross_project(
        self,
        query: str,
        limit: int = 5,
        include_current: bool = True,
        min_similarity: float = 0.3,
    ) -> list[CrossProjectMatch]:
        """
        Search for relevant memories across all projects.

        Args:
            query: Search query
            limit: Maximum results
            include_current: Include current project in results
            min_similarity: Minimum similarity threshold

        Returns:
            List of cross-project matches with relevance scores
        """
        service = get_embedding_service()
        query_embedding = service.embed(query).embedding

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        # Get all successful, transferable memories
        if include_current:
            cursor.execute("""
                SELECT pm.id, pm.project_id, pm.task, pm.solution, pm.embedding, p.name
                FROM project_memories pm
                JOIN projects p ON pm.project_id = p.id
                WHERE pm.success = 1
            """)
        else:
            cursor.execute("""
                SELECT pm.id, pm.project_id, pm.task, pm.solution, pm.embedding, p.name
                FROM project_memories pm
                JOIN projects p ON pm.project_id = p.id
                WHERE pm.success = 1 AND pm.project_id != ?
            """, (self._current_project,))

        rows = cursor.fetchall()
        conn.close()

        matches = []
        for row in rows:
            memory_id, project_id, task, solution, embedding_json, project_name = row
            stored_embedding = json.loads(embedding_json) if embedding_json else []

            similarity = service.similarity(query_embedding, stored_embedding)

            if similarity >= min_similarity:
                # Adjust relevance based on whether it's from current project
                if project_id == self._current_project:
                    relevance = similarity  # Full weight for current project
                else:
                    # Cross-project transfer gets slight penalty but still valuable
                    relevance = similarity * 0.85

                matches.append(CrossProjectMatch(
                    memory_id=memory_id,
                    task=task,
                    solution=solution,
                    source_project=project_id,
                    source_project_name=project_name,
                    similarity=similarity,
                    relevance_score=relevance,
                ))

        # Sort by relevance
        matches.sort(key=lambda m: m.relevance_score, reverse=True)
        return matches[:limit]

    def get_universal_insights(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.4,
    ) -> list[UniversalInsight]:
        """
        Get universal insights relevant to a query.

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant universal insights
        """
        service = get_embedding_service()
        query_embedding = service.embed(query).embedding

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, pattern, insight, source_projects, success_count, apply_count, embedding
            FROM universal_insights
        """)

        rows = cursor.fetchall()
        conn.close()

        insights = []
        for row in rows:
            stored_embedding = json.loads(row[6]) if row[6] else []
            similarity = service.similarity(query_embedding, stored_embedding)

            if similarity >= min_similarity:
                insights.append(UniversalInsight(
                    id=row[0],
                    pattern=row[1],
                    insight=row[2],
                    source_projects=json.loads(row[3]) if row[3] else [],
                    success_count=row[4] or 0,
                    apply_count=row[5] or 0,
                    embedding=stored_embedding,
                ))

        # Sort by effectiveness and similarity
        insights.sort(key=lambda i: (i.effectiveness, similarity), reverse=True)
        return insights[:limit]

    def record_transfer(
        self,
        source_project: str,
        target_project: str,
        memory_id: Optional[str] = None,
        insight_id: Optional[str] = None,
        success: bool = True,
    ):
        """Record a knowledge transfer event."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO transfer_log (source_project, target_project, memory_id, insight_id, success, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            source_project,
            target_project,
            memory_id,
            insight_id,
            1 if success else 0,
            datetime.now().isoformat(),
        ))

        # Update transfer count on memory
        if memory_id:
            cursor.execute("""
                UPDATE project_memories SET transfer_count = transfer_count + 1 WHERE id = ?
            """, (memory_id,))

        # Update apply count on insight
        if insight_id:
            if success:
                cursor.execute("""
                    UPDATE universal_insights
                    SET apply_count = apply_count + 1, success_count = success_count + 1
                    WHERE id = ?
                """, (insight_id,))
            else:
                cursor.execute("""
                    UPDATE universal_insights SET apply_count = apply_count + 1 WHERE id = ?
                """, (insight_id,))

        conn.commit()
        conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge transfer statistics."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM projects")
        total_projects = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM project_memories")
        total_memories = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM project_memories WHERE is_transferable = 1")
        transferable_memories = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM universal_insights")
        total_insights = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM transfer_log")
        total_transfers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM transfer_log WHERE success = 1")
        successful_transfers = cursor.fetchone()[0]

        conn.close()

        return {
            "total_projects": total_projects,
            "total_memories": total_memories,
            "transferable_memories": transferable_memories,
            "universal_insights": total_insights,
            "total_transfers": total_transfers,
            "successful_transfers": successful_transfers,
            "transfer_success_rate": successful_transfers / total_transfers if total_transfers > 0 else 0,
        }

    def list_projects(self) -> list[ProjectInfo]:
        """List all known projects."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, path, language, framework, created_at, last_accessed
            FROM projects ORDER BY last_accessed DESC
        """)

        projects = []
        for row in cursor.fetchall():
            projects.append(ProjectInfo(
                id=row[0],
                name=row[1],
                path=row[2],
                language=row[3] or "",
                framework=row[4] or "",
                created_at=datetime.fromisoformat(row[5]),
                last_accessed=datetime.fromisoformat(row[6]),
            ))

        conn.close()
        return projects


# Global singleton
_knowledge_store: Optional[KnowledgeStore] = None


def get_knowledge_store() -> KnowledgeStore:
    """Get or create the global knowledge store."""
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = KnowledgeStore()
    return _knowledge_store
