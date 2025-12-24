"""
Hierarchical Goal Decomposition: Recursive planning and execution.

Enables breaking complex goals into nested hierarchies of sub-goals,
creating tree-structured plans that can be executed bottom-up.

Architecture:
                    [Root Goal]
                   /    |    \\
            [Sub-Goal] [Sub-Goal] [Sub-Goal]
              /  \\        |
        [Task]  [Task]  [Task]

Execution flows bottom-up: leaves execute first, results propagate up.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
import time

from agents.grunt import GruntAgent, GruntResult
from router import ModelRouter


class GoalStatus(Enum):
    """Status of a goal in the hierarchy."""
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Goal:
    """
    A node in the goal hierarchy tree.

    Goals can contain sub-goals (children), forming a recursive structure.
    Leaf goals are atomic tasks that can be executed directly.
    """
    id: str
    description: str
    depth: int = 0
    status: GoalStatus = GoalStatus.PENDING
    complexity: str = "medium"  # low, medium, high
    children: list["Goal"] = field(default_factory=list)
    parent_id: str | None = None

    # Execution state
    result: str | None = None
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    tokens_used: int = 0
    model_used: str | None = None

    # Planning metadata
    rationale: str | None = None  # Why this decomposition
    success_criteria: str | None = None  # How to know it's done

    def is_leaf(self) -> bool:
        """Check if this goal is a leaf (atomic task)."""
        return len(self.children) == 0

    def is_complete(self) -> bool:
        """Check if goal is complete (self or all children)."""
        if self.is_leaf():
            return self.status == GoalStatus.COMPLETED
        return all(child.is_complete() for child in self.children)

    def get_ready_children(self) -> list["Goal"]:
        """Get children that are ready to execute (pending with no blockers)."""
        ready = []
        completed_ids = {c.id for c in self.children if c.status == GoalStatus.COMPLETED}

        for child in self.children:
            if child.status == GoalStatus.PENDING:
                # Check if any sibling dependencies are met
                # For now, assume all siblings are independent
                ready.append(child)

        return ready

    def get_depth_first_leaves(self) -> list["Goal"]:
        """Get all leaf goals in depth-first order."""
        if self.is_leaf():
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_depth_first_leaves())
        return leaves

    def get_progress(self) -> tuple[int, int]:
        """Get (completed, total) count of leaf goals."""
        leaves = self.get_depth_first_leaves()
        completed = sum(1 for l in leaves if l.status == GoalStatus.COMPLETED)
        return completed, len(leaves)

    def to_dict(self) -> dict:
        """Serialize goal to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "depth": self.depth,
            "status": self.status.value,
            "complexity": self.complexity,
            "children": [c.to_dict() for c in self.children],
            "parent_id": self.parent_id,
            "result": self.result,
            "error": self.error,
            "rationale": self.rationale,
            "success_criteria": self.success_criteria,
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        """Deserialize goal from dictionary."""
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            id=data["id"],
            description=data["description"],
            depth=data.get("depth", 0),
            status=GoalStatus(data.get("status", "pending")),
            complexity=data.get("complexity", "medium"),
            children=children,
            parent_id=data.get("parent_id"),
            result=data.get("result"),
            error=data.get("error"),
            rationale=data.get("rationale"),
            success_criteria=data.get("success_criteria"),
            tokens_used=data.get("tokens_used", 0),
        )


@dataclass
class GoalTree:
    """
    Complete goal hierarchy with execution tracking.
    """
    root: Goal
    created_at: datetime = field(default_factory=datetime.now)
    total_tokens: int = 0
    planning_tokens: int = 0
    execution_tokens: int = 0

    def get_all_goals(self) -> list[Goal]:
        """Get all goals in the tree (BFS order)."""
        goals = []
        queue = [self.root]
        while queue:
            goal = queue.pop(0)
            goals.append(goal)
            queue.extend(goal.children)
        return goals

    def find_goal(self, goal_id: str) -> Goal | None:
        """Find a goal by ID."""
        for goal in self.get_all_goals():
            if goal.id == goal_id:
                return goal
        return None

    def get_status_summary(self) -> dict[str, int]:
        """Get count of goals by status."""
        summary = {s.value: 0 for s in GoalStatus}
        for goal in self.get_all_goals():
            summary[goal.status.value] += 1
        return summary

    def is_complete(self) -> bool:
        """Check if entire tree is complete."""
        return self.root.is_complete()

    def get_next_executable(self) -> list[Goal]:
        """Get the next batch of executable leaf goals."""
        executable = []
        for goal in self.get_all_goals():
            if goal.is_leaf() and goal.status == GoalStatus.PENDING:
                executable.append(goal)
        return executable


PLANNER_SYSTEM_PROMPT = """You are a hierarchical goal planner. Your task is to decompose complex goals into simpler sub-goals.

Rules:
1. Only decompose if the goal is too complex to execute directly
2. Each sub-goal should be simpler than its parent
3. Sub-goals should be independent when possible (parallelizable)
4. Provide clear success criteria for verification
5. Stop decomposing when goals become atomic tasks

Respond in this exact JSON format:
{
    "should_decompose": true/false,
    "rationale": "Why decompose or not",
    "complexity": "low/medium/high",
    "sub_goals": [
        {
            "description": "Clear sub-goal description",
            "complexity": "low/medium/high",
            "success_criteria": "How to verify completion"
        }
    ]
}

If should_decompose is false, sub_goals should be empty."""


class HierarchicalPlanner:
    """
    Plans and executes hierarchical goal structures.

    Capabilities:
    - Recursive decomposition with depth limits
    - Complexity-based decomposition decisions
    - Bottom-up execution with result propagation
    - Dynamic re-planning on failure
    """

    # Configuration
    MAX_DEPTH = 3  # Maximum decomposition depth
    MAX_CHILDREN = 5  # Maximum sub-goals per goal
    MIN_COMPLEXITY_TO_DECOMPOSE = "medium"  # Only decompose medium+ complexity

    def __init__(self, working_dir: str | None = None, db_path: str | None = None):
        self.working_dir = working_dir
        self.router = ModelRouter()

        # Database for persistence
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "goals.db")

        self.db_path = db_path
        self._init_db()

        # Callbacks
        self.on_decompose: Callable[[Goal, list[Goal]], None] | None = None
        self.on_goal_start: Callable[[Goal], None] | None = None
        self.on_goal_complete: Callable[[Goal], None] | None = None
        self.on_progress: Callable[[int, int], None] | None = None

    def _init_db(self):
        """Initialize goal tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goal_trees (
                id TEXT PRIMARY KEY,
                root_description TEXT NOT NULL,
                tree_json TEXT NOT NULL,
                status TEXT NOT NULL,
                total_goals INTEGER,
                completed_goals INTEGER,
                total_tokens INTEGER,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trees_created
            ON goal_trees(created_at DESC)
        """)

        conn.commit()
        conn.close()

    def plan(self, task: str, context: str = "") -> GoalTree:
        """
        Create a hierarchical plan for a task.

        Args:
            task: The high-level goal to achieve
            context: Additional context

        Returns:
            GoalTree with decomposed hierarchy
        """
        # Create root goal
        root = Goal(
            id="goal_0",
            description=task,
            depth=0,
        )

        tree = GoalTree(root=root)

        # Recursively decompose
        self._decompose_recursive(root, context, tree)

        return tree

    def _decompose_recursive(
        self,
        goal: Goal,
        context: str,
        tree: GoalTree,
        counter: list[int] | None = None,
    ):
        """Recursively decompose a goal into sub-goals."""
        if counter is None:
            counter = [1]  # Mutable counter for ID generation

        # Check depth limit
        if goal.depth >= self.MAX_DEPTH:
            return

        # Analyze if decomposition is needed
        analysis = self._analyze_goal(goal, context)
        tree.planning_tokens += analysis.get("tokens_used", 0)
        tree.total_tokens += analysis.get("tokens_used", 0)

        goal.complexity = analysis.get("complexity", "medium")
        goal.rationale = analysis.get("rationale", "")

        if not analysis.get("should_decompose", False):
            return

        # Create sub-goals
        sub_goals_data = analysis.get("sub_goals", [])[:self.MAX_CHILDREN]

        for sg_data in sub_goals_data:
            sub_goal = Goal(
                id=f"goal_{counter[0]}",
                description=sg_data.get("description", ""),
                depth=goal.depth + 1,
                complexity=sg_data.get("complexity", "medium"),
                parent_id=goal.id,
                success_criteria=sg_data.get("success_criteria"),
            )
            counter[0] += 1
            goal.children.append(sub_goal)

            # Recursively decompose if still complex
            if sub_goal.complexity in ("medium", "high"):
                self._decompose_recursive(sub_goal, context, tree, counter)

        if self.on_decompose:
            self.on_decompose(goal, goal.children)

    def _analyze_goal(self, goal: Goal, context: str) -> dict:
        """Analyze whether a goal needs decomposition."""
        from agents.base import BaseAgent, Message

        agent = BaseAgent(model="sonnet", system_prompt=PLANNER_SYSTEM_PROMPT)

        user_content = f"## Goal\n{goal.description}\n\n"
        user_content += f"## Current Depth\n{goal.depth} (max: {self.MAX_DEPTH})\n\n"

        if context:
            user_content += f"## Context\n{context}\n\n"

        user_content += "Analyze this goal and decide if it should be decomposed into sub-goals."

        messages = [
            Message(role="system", content=PLANNER_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        response = agent._call_llm(messages)

        try:
            # Extract JSON
            text = response.content
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())
            result["tokens_used"] = response.input_tokens + response.output_tokens
            return result

        except (json.JSONDecodeError, KeyError):
            return {
                "should_decompose": False,
                "rationale": "Failed to analyze goal",
                "complexity": "medium",
                "sub_goals": [],
                "tokens_used": response.input_tokens + response.output_tokens,
            }

    def execute(
        self,
        tree: GoalTree,
        context: str = "",
        parallel: bool = True,
    ) -> GoalTree:
        """
        Execute a goal tree from leaves to root.

        Args:
            tree: The goal tree to execute
            context: Additional context
            parallel: Execute independent goals in parallel

        Returns:
            Updated goal tree with results
        """
        import asyncio

        if parallel:
            asyncio.run(self._execute_parallel(tree, context))
        else:
            self._execute_sequential(tree, context)

        return tree

    def _execute_sequential(self, tree: GoalTree, context: str):
        """Execute goals sequentially, leaves first."""
        while not tree.is_complete():
            # Get next batch of executable goals
            executable = tree.get_next_executable()

            if not executable:
                # Check for blocked goals
                blocked = [g for g in tree.get_all_goals()
                          if g.status == GoalStatus.PENDING and not g.is_leaf()]
                if blocked:
                    # Parent goals waiting on failed children
                    for goal in blocked:
                        if any(c.status == GoalStatus.FAILED for c in goal.children):
                            goal.status = GoalStatus.FAILED
                            goal.error = "Child goal failed"
                break

            for goal in executable:
                self._execute_single_goal(goal, context, tree)

                # Update progress
                completed, total = tree.root.get_progress()
                if self.on_progress:
                    self.on_progress(completed, total)

            # After executing leaves, check if parents are now complete
            self._propagate_completion(tree)

    async def _execute_parallel(self, tree: GoalTree, context: str):
        """Execute independent goals in parallel."""
        while not tree.is_complete():
            executable = tree.get_next_executable()

            if not executable:
                blocked = [g for g in tree.get_all_goals()
                          if g.status == GoalStatus.PENDING and not g.is_leaf()]
                if blocked:
                    for goal in blocked:
                        if any(c.status == GoalStatus.FAILED for c in goal.children):
                            goal.status = GoalStatus.FAILED
                            goal.error = "Child goal failed"
                break

            # Execute batch in parallel
            tasks = [
                self._execute_single_goal_async(goal, context, tree)
                for goal in executable
            ]
            await asyncio.gather(*tasks)

            # Update progress
            completed, total = tree.root.get_progress()
            if self.on_progress:
                self.on_progress(completed, total)

            self._propagate_completion(tree)

    def _execute_single_goal(self, goal: Goal, context: str, tree: GoalTree):
        """Execute a single leaf goal."""
        goal.status = GoalStatus.IN_PROGRESS
        goal.start_time = time.time()

        if self.on_goal_start:
            self.on_goal_start(goal)

        try:
            # Select model based on complexity
            model = self.router.select(goal.description, complexity=goal.complexity)
            goal.model_used = model

            # Build context from parent chain
            parent_context = self._build_parent_context(goal, tree)
            full_context = f"{context}\n\n{parent_context}" if parent_context else context

            # Add success criteria if available
            task = goal.description
            if goal.success_criteria:
                task += f"\n\nSuccess criteria: {goal.success_criteria}"

            # Execute
            grunt = GruntAgent(model=model, working_dir=self.working_dir)
            result = grunt.run(task, full_context)

            goal.tokens_used = result.input_tokens + result.output_tokens
            tree.execution_tokens += goal.tokens_used
            tree.total_tokens += goal.tokens_used

            if result.success:
                goal.status = GoalStatus.COMPLETED
                goal.result = result.result
            else:
                goal.status = GoalStatus.FAILED
                goal.error = result.error

        except Exception as e:
            goal.status = GoalStatus.FAILED
            goal.error = str(e)

        goal.end_time = time.time()

        if self.on_goal_complete:
            self.on_goal_complete(goal)

    async def _execute_single_goal_async(self, goal: Goal, context: str, tree: GoalTree):
        """Async version of single goal execution."""
        # For now, wrap synchronous execution
        # Could be enhanced with true async LLM calls
        self._execute_single_goal(goal, context, tree)

    def _build_parent_context(self, goal: Goal, tree: GoalTree) -> str:
        """Build context from completed sibling and parent results."""
        parts = []

        # Get parent chain
        parent = tree.find_goal(goal.parent_id) if goal.parent_id else None

        if parent:
            parts.append(f"## Parent Goal\n{parent.description}")

            # Add completed siblings' results
            completed_siblings = [
                c for c in parent.children
                if c.id != goal.id and c.status == GoalStatus.COMPLETED
            ]

            if completed_siblings:
                parts.append("\n## Completed Sibling Results")
                for sibling in completed_siblings:
                    parts.append(f"\n### {sibling.description[:100]}")
                    if sibling.result:
                        parts.append(sibling.result[:500])

        return "\n".join(parts)

    def _propagate_completion(self, tree: GoalTree):
        """Propagate completion status from leaves to parents."""
        # Work from deepest to shallowest
        all_goals = tree.get_all_goals()
        by_depth = {}
        for goal in all_goals:
            by_depth.setdefault(goal.depth, []).append(goal)

        max_depth = max(by_depth.keys()) if by_depth else 0

        for depth in range(max_depth, -1, -1):
            for goal in by_depth.get(depth, []):
                if not goal.is_leaf() and goal.status not in (GoalStatus.COMPLETED, GoalStatus.FAILED):
                    if all(c.status == GoalStatus.COMPLETED for c in goal.children):
                        goal.status = GoalStatus.COMPLETED
                        # Aggregate child results
                        results = [c.result for c in goal.children if c.result]
                        goal.result = "\n\n---\n\n".join(results) if results else "All sub-goals completed"
                    elif any(c.status == GoalStatus.FAILED for c in goal.children):
                        goal.status = GoalStatus.FAILED
                        failed = [c for c in goal.children if c.status == GoalStatus.FAILED]
                        goal.error = f"{len(failed)} sub-goal(s) failed"

    def save_tree(self, tree: GoalTree) -> str:
        """Save goal tree to database."""
        tree_id = f"tree_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        completed, total = tree.root.get_progress()
        status = "completed" if tree.is_complete() else "in_progress"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO goal_trees
            (id, root_description, tree_json, status, total_goals,
             completed_goals, total_tokens, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tree_id,
            tree.root.description[:500],
            json.dumps(tree.root.to_dict()),
            status,
            total,
            completed,
            tree.total_tokens,
            tree.created_at.isoformat(),
            datetime.now().isoformat() if tree.is_complete() else None,
        ))

        conn.commit()
        conn.close()

        return tree_id

    def load_tree(self, tree_id: str) -> GoalTree | None:
        """Load goal tree from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT tree_json, total_tokens, created_at FROM goal_trees WHERE id = ?",
            (tree_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        tree_json, total_tokens, created_at = row
        root = Goal.from_dict(json.loads(tree_json))

        tree = GoalTree(
            root=root,
            total_tokens=total_tokens,
            created_at=datetime.fromisoformat(created_at),
        )

        return tree

    def get_stats(self) -> dict:
        """Get hierarchical planning statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM goal_trees")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM goal_trees WHERE status = 'completed'")
        completed = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(total_goals) FROM goal_trees")
        avg_goals = cursor.fetchone()[0] or 0

        cursor.execute("SELECT AVG(total_tokens) FROM goal_trees")
        avg_tokens = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_plans": total,
            "completed_plans": completed,
            "success_rate": completed / total if total > 0 else 0,
            "avg_goals_per_plan": avg_goals,
            "avg_tokens_per_plan": avg_tokens,
        }

    def format_tree(self, tree: GoalTree) -> str:
        """Format goal tree for display."""
        lines = ["## Goal Hierarchy\n"]

        def format_goal(goal: Goal, indent: int = 0):
            prefix = "  " * indent
            status_icon = {
                GoalStatus.PENDING: "[ ]",
                GoalStatus.PLANNING: "[~]",
                GoalStatus.IN_PROGRESS: "[>]",
                GoalStatus.COMPLETED: "[x]",
                GoalStatus.FAILED: "[!]",
                GoalStatus.BLOCKED: "[#]",
            }.get(goal.status, "[ ]")

            lines.append(f"{prefix}{status_icon} {goal.description[:80]}")

            if goal.error:
                lines.append(f"{prefix}    Error: {goal.error[:100]}")

            for child in goal.children:
                format_goal(child, indent + 1)

        format_goal(tree.root)

        completed, total = tree.root.get_progress()
        lines.append(f"\nProgress: {completed}/{total} tasks completed")
        lines.append(f"Total tokens: {tree.total_tokens:,}")

        return "\n".join(lines)


# Global instance
_planner: HierarchicalPlanner | None = None


def get_hierarchical_planner(working_dir: str | None = None) -> HierarchicalPlanner:
    """Get or create the global hierarchical planner."""
    global _planner
    if _planner is None:
        _planner = HierarchicalPlanner(working_dir=working_dir)
    return _planner
