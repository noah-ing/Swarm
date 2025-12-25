"""
Rollback Planning: Capture state and revert changes if needed.

Provides:
- File state snapshots before changes
- Rollback plan generation
- Safe rollback execution
- Git integration for version control
"""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class FileSnapshot:
    """Snapshot of a file's state."""
    path: str
    content: str
    hash: str
    exists: bool
    permissions: int = 0o644


@dataclass
class RollbackPlan:
    """A plan for rolling back changes."""
    id: str
    task: str
    created_at: datetime
    snapshots: list[FileSnapshot]
    files_to_delete: list[str] = field(default_factory=list)  # New files created
    git_commit_before: str = ""  # Git commit hash before changes
    executed: bool = False
    rolled_back: bool = False


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    files_restored: list[str]
    files_deleted: list[str]
    errors: list[str]
    used_git: bool = False


class RollbackManager:
    """
    Manages rollback plans for safe change execution.

    Captures file states before changes and provides
    the ability to restore them if something goes wrong.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "rollback.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the rollback database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rollback_plans (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                root_path TEXT NOT NULL,
                snapshots TEXT NOT NULL,
                files_to_delete TEXT,
                git_commit_before TEXT,
                executed INTEGER DEFAULT 0,
                rolled_back INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rollback_created
            ON rollback_plans(created_at DESC)
        """)

        conn.commit()
        conn.close()

    def _hash_content(self, content: str) -> str:
        """Create hash of file content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_git_head(self, root_path: str) -> str:
        """Get current git HEAD commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def _is_git_repo(self, root_path: str) -> bool:
        """Check if path is in a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=root_path,
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def create_plan(
        self,
        task: str,
        root_path: str,
        files_to_watch: list[str],
    ) -> RollbackPlan:
        """
        Create a rollback plan by snapshotting specified files.

        Args:
            task: Description of the task being performed
            root_path: Root path of the project
            files_to_watch: Files that might be modified

        Returns:
            RollbackPlan with file snapshots
        """
        plan_id = hashlib.sha256(
            f"{task}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        snapshots = []
        for file_path in files_to_watch:
            # Handle both relative and absolute paths
            if not os.path.isabs(file_path):
                full_path = os.path.join(root_path, file_path)
            else:
                full_path = file_path

            snapshot = self._snapshot_file(full_path)
            if snapshot:
                # Store relative path
                rel_path = os.path.relpath(full_path, root_path)
                snapshot.path = rel_path
                snapshots.append(snapshot)

        # Get git state
        git_commit = self._get_git_head(root_path) if self._is_git_repo(root_path) else ""

        plan = RollbackPlan(
            id=plan_id,
            task=task,
            created_at=datetime.now(),
            snapshots=snapshots,
            git_commit_before=git_commit,
        )

        # Store plan
        self._store_plan(plan, root_path)

        return plan

    def _snapshot_file(self, file_path: str) -> Optional[FileSnapshot]:
        """Create a snapshot of a file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                stat = os.stat(file_path)
                return FileSnapshot(
                    path=file_path,
                    content=content,
                    hash=self._hash_content(content),
                    exists=True,
                    permissions=stat.st_mode,
                )
            else:
                # File doesn't exist - snapshot the absence
                return FileSnapshot(
                    path=file_path,
                    content="",
                    hash="",
                    exists=False,
                )
        except Exception:
            return None

    def _store_plan(self, plan: RollbackPlan, root_path: str):
        """Store rollback plan in database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        snapshots_json = json.dumps([
            {
                "path": s.path,
                "content": s.content,
                "hash": s.hash,
                "exists": s.exists,
                "permissions": s.permissions,
            }
            for s in plan.snapshots
        ])

        cursor.execute("""
            INSERT INTO rollback_plans
            (id, task, root_path, snapshots, files_to_delete, git_commit_before, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            plan.id,
            plan.task,
            root_path,
            snapshots_json,
            json.dumps(plan.files_to_delete),
            plan.git_commit_before,
            plan.created_at.isoformat(),
        ))

        conn.commit()
        conn.close()

    def mark_executed(self, plan_id: str, new_files: Optional[list[str]] = None):
        """Mark a plan as executed, optionally recording new files created."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        files_to_delete = json.dumps(new_files or [])

        cursor.execute("""
            UPDATE rollback_plans
            SET executed = 1, files_to_delete = ?
            WHERE id = ?
        """, (files_to_delete, plan_id))

        conn.commit()
        conn.close()

    def rollback(
        self,
        plan_id: str,
        use_git: bool = True,
    ) -> RollbackResult:
        """
        Execute a rollback to restore files to their previous state.

        Args:
            plan_id: ID of the rollback plan
            use_git: Whether to use git reset if available

        Returns:
            RollbackResult with details of what was restored
        """
        # Load plan
        plan, root_path = self._load_plan(plan_id)
        if not plan:
            return RollbackResult(
                success=False,
                files_restored=[],
                files_deleted=[],
                errors=["Rollback plan not found"],
            )

        if plan.rolled_back:
            return RollbackResult(
                success=False,
                files_restored=[],
                files_deleted=[],
                errors=["Plan has already been rolled back"],
            )

        files_restored = []
        files_deleted = []
        errors = []
        used_git = False

        # Try git reset first if available and requested
        if use_git and plan.git_commit_before and self._is_git_repo(root_path):
            try:
                # Check if we have uncommitted changes
                status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=root_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if status.returncode == 0:
                    # Reset to previous commit
                    result = subprocess.run(
                        ["git", "checkout", plan.git_commit_before, "--", "."],
                        cwd=root_path,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    if result.returncode == 0:
                        used_git = True
                        files_restored = [s.path for s in plan.snapshots if s.exists]
            except Exception as e:
                errors.append(f"Git rollback failed: {e}")

        # If git didn't work, restore files manually
        if not used_git:
            for snapshot in plan.snapshots:
                try:
                    full_path = os.path.join(root_path, snapshot.path)

                    if snapshot.exists:
                        # Restore file content
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(snapshot.content)
                        os.chmod(full_path, snapshot.permissions)
                        files_restored.append(snapshot.path)
                    else:
                        # File didn't exist before - delete it
                        if os.path.exists(full_path):
                            os.remove(full_path)
                            files_deleted.append(snapshot.path)
                except Exception as e:
                    errors.append(f"Failed to restore {snapshot.path}: {e}")

        # Delete new files that were created
        for file_path in plan.files_to_delete:
            try:
                full_path = os.path.join(root_path, file_path)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    files_deleted.append(file_path)
            except Exception as e:
                errors.append(f"Failed to delete {file_path}: {e}")

        # Mark as rolled back
        self._mark_rolled_back(plan_id)

        return RollbackResult(
            success=len(errors) == 0,
            files_restored=files_restored,
            files_deleted=files_deleted,
            errors=errors,
            used_git=used_git,
        )

    def _load_plan(self, plan_id: str) -> tuple[Optional[RollbackPlan], str]:
        """Load a rollback plan from database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT task, root_path, snapshots, files_to_delete, git_commit_before,
                   executed, rolled_back, created_at
            FROM rollback_plans WHERE id = ?
        """, (plan_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None, ""

        snapshots_data = json.loads(row[2])
        snapshots = [
            FileSnapshot(
                path=s["path"],
                content=s["content"],
                hash=s["hash"],
                exists=s["exists"],
                permissions=s.get("permissions", 0o644),
            )
            for s in snapshots_data
        ]

        plan = RollbackPlan(
            id=plan_id,
            task=row[0],
            created_at=datetime.fromisoformat(row[7]),
            snapshots=snapshots,
            files_to_delete=json.loads(row[3]) if row[3] else [],
            git_commit_before=row[4] or "",
            executed=bool(row[5]),
            rolled_back=bool(row[6]),
        )

        return plan, row[1]

    def _mark_rolled_back(self, plan_id: str):
        """Mark a plan as rolled back."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE rollback_plans SET rolled_back = 1 WHERE id = ?
        """, (plan_id,))

        conn.commit()
        conn.close()

    def get_recent_plans(self, limit: int = 10) -> list[dict]:
        """Get recent rollback plans."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, task, root_path, executed, rolled_back, created_at
            FROM rollback_plans
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        plans = []
        for row in cursor.fetchall():
            plans.append({
                "id": row[0],
                "task": row[1][:50],
                "root_path": row[2],
                "executed": bool(row[3]),
                "rolled_back": bool(row[4]),
                "created_at": row[5],
            })

        conn.close()
        return plans

    def cleanup_old_plans(self, days: int = 7) -> int:
        """Remove rollback plans older than specified days."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()

        cursor.execute("""
            DELETE FROM rollback_plans WHERE created_at < ?
        """, (cutoff_iso,))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted

    def get_stats(self) -> dict:
        """Get rollback statistics."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM rollback_plans")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM rollback_plans WHERE executed = 1")
        executed = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM rollback_plans WHERE rolled_back = 1")
        rolled_back = cursor.fetchone()[0]

        conn.close()

        return {
            "total_plans": total,
            "executed": executed,
            "rolled_back": rolled_back,
            "rollback_rate": rolled_back / executed if executed > 0 else 0,
        }


# Global instance
_rollback_manager: Optional[RollbackManager] = None


def get_rollback_manager() -> RollbackManager:
    """Get or create the global rollback manager."""
    global _rollback_manager
    if _rollback_manager is None:
        _rollback_manager = RollbackManager()
    return _rollback_manager
