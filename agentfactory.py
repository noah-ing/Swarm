"""
Agent Factory: Self-Modifying Architecture for Swarm.

This module enables Swarm to create, evolve, and retire agent types dynamically.
The system can literally write new agents based on observed needs.

Capabilities:
1. Analyze task patterns to identify gaps in agent coverage
2. Generate new agent classes from templates
3. Test agents in sandbox before deployment
4. Evolve agent architectures based on performance
5. Retire underperforming agents

This is self-modifying code - Swarm evolves its own structure.
"""

import ast
import sqlite3
import json
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from enum import Enum
import textwrap


class AgentRole(Enum):
    """Roles that agents can fulfill."""
    EXECUTOR = "executor"  # Runs tasks directly
    ANALYZER = "analyzer"  # Analyzes code/data
    VALIDATOR = "validator"  # Validates outputs
    TRANSFORMER = "transformer"  # Transforms data/code
    SPECIALIST = "specialist"  # Domain-specific expert


@dataclass
class AgentBlueprint:
    """Blueprint for a dynamically created agent."""
    id: str
    name: str
    role: AgentRole
    description: str
    system_prompt: str
    capabilities: list[str]
    tools: list[str]  # Tools this agent can use
    parent_id: str | None = None  # If evolved from another agent
    generation: int = 0

    # Performance tracking
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_tokens: float = 0.0
    avg_duration: float = 0.0

    # Lifecycle
    status: str = "active"  # active, testing, retired
    created_at: datetime = field(default_factory=datetime.now)
    retired_at: datetime | None = None


@dataclass
class AgentNeed:
    """Identified need for a new agent type."""
    pattern: str  # Task pattern that needs coverage
    frequency: int  # How often this pattern occurs
    current_success_rate: float  # Success rate with existing agents
    suggested_role: AgentRole
    suggested_capabilities: list[str]
    confidence: float


# Agent templates for different roles
AGENT_TEMPLATES = {
    AgentRole.EXECUTOR: '''
class {class_name}(BaseAgent):
    """
    {description}

    Auto-generated agent for: {capabilities}
    Generation: {generation}
    """

    def __init__(self, model: str = "sonnet"):
        super().__init__(model=model)
        self.system_prompt = """{system_prompt}"""

    def run(self, task: str, context: str = "") -> dict:
        """Execute the task."""
        messages = [
            {{"role": "system", "content": self.system_prompt}},
            {{"role": "user", "content": f"{{task}}\\n\\nContext: {{context}}"}}
        ]

        response = self.chat(messages=messages)

        return {{
            "success": True,
            "result": response.content,
            "tokens": response.input_tokens + response.output_tokens,
        }}
''',

    AgentRole.ANALYZER: '''
class {class_name}(BaseAgent):
    """
    {description}

    Auto-generated analyzer for: {capabilities}
    Generation: {generation}
    """

    def __init__(self, model: str = "sonnet"):
        super().__init__(model=model)
        self.system_prompt = """{system_prompt}"""

    def run(self, task: str, context: str = "") -> dict:
        """Run analysis task."""
        return self.analyze(task, context)

    def analyze(self, content: str, focus: str = "") -> dict:
        """Analyze content with optional focus area."""
        prompt = f"Analyze the following:\\n\\n{{content}}"
        if focus:
            prompt += f"\\n\\nFocus on: {{focus}}"

        messages = [
            {{"role": "system", "content": self.system_prompt}},
            {{"role": "user", "content": prompt}}
        ]

        response = self.chat(messages=messages)

        return {{
            "result": response.content,
            "analysis": response.content,
            "tokens": response.input_tokens + response.output_tokens,
        }}
''',

    AgentRole.VALIDATOR: '''
class {class_name}(BaseAgent):
    """
    {description}

    Auto-generated validator for: {capabilities}
    Generation: {generation}
    """

    def __init__(self, model: str = "sonnet"):
        super().__init__(model=model)
        self.system_prompt = """{system_prompt}"""

    def run(self, task: str, context: str = "") -> dict:
        """Run validation task."""
        return self.validate(task, context.split(",") if context else None)

    def validate(self, content: str, criteria: list[str] = None) -> dict:
        """Validate content against criteria."""
        prompt = f"Validate the following:\\n\\n{{content}}"
        if criteria:
            prompt += f"\\n\\nCriteria: {{', '.join(criteria)}}"

        messages = [
            {{"role": "system", "content": self.system_prompt}},
            {{"role": "user", "content": prompt}}
        ]

        response = self.chat(messages=messages)

        # Parse validation result
        is_valid = "invalid" not in response.content.lower()[:100]

        return {{
            "valid": is_valid,
            "result": response.content,
            "feedback": response.content,
            "tokens": response.input_tokens + response.output_tokens,
        }}
''',

    AgentRole.TRANSFORMER: '''
class {class_name}(BaseAgent):
    """
    {description}

    Auto-generated transformer for: {capabilities}
    Generation: {generation}
    """

    def __init__(self, model: str = "sonnet"):
        super().__init__(model=model)
        self.system_prompt = """{system_prompt}"""

    def run(self, task: str, context: str = "") -> dict:
        """Run transformation task."""
        return self.transform(task, context or "requested format")

    def transform(self, input_content: str, target_format: str) -> dict:
        """Transform content to target format."""
        prompt = f"Transform the following to {{target_format}}:\\n\\n{{input_content}}"

        messages = [
            {{"role": "system", "content": self.system_prompt}},
            {{"role": "user", "content": prompt}}
        ]

        response = self.chat(messages=messages)

        return {{
            "result": response.content,
            "transformed": response.content,
            "tokens": response.input_tokens + response.output_tokens,
        }}
''',

    AgentRole.SPECIALIST: '''
class {class_name}(BaseAgent):
    """
    {description}

    Auto-generated specialist for: {capabilities}
    Generation: {generation}
    """

    def __init__(self, model: str = "sonnet"):
        super().__init__(model=model)
        self.system_prompt = """{system_prompt}"""
        self.specialty = "{specialty}"

    def run(self, task: str, context: str = "") -> dict:
        """Run specialist consultation."""
        return self.consult(task, context)

    def consult(self, question: str, context: str = "") -> dict:
        """Provide specialist consultation."""
        prompt = f"As a {{self.specialty}} specialist:\\n\\n{{question}}"
        if context:
            prompt += f"\\n\\nContext: {{context}}"

        messages = [
            {{"role": "system", "content": self.system_prompt}},
            {{"role": "user", "content": prompt}}
        ]

        response = self.chat(messages=messages)

        return {{
            "result": response.content,
            "advice": response.content,
            "tokens": response.input_tokens + response.output_tokens,
        }}
''',
}


class AgentFactory:
    """
    Factory for creating, evolving, and managing dynamic agents.

    This is self-modifying architecture - the system creates new agent types
    based on observed needs and performance.
    """

    # Thresholds
    MIN_PATTERN_FREQUENCY = 3  # Min occurrences before creating agent
    MIN_SUCCESS_RATE_GAP = 0.2  # Min gap to justify new agent
    RETIREMENT_THRESHOLD = 0.3  # Retire agents below this success rate

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path.home() / ".swarm"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "agentfactory.db")

        self.db_path = db_path
        self._init_db()

        # Cache of loaded dynamic agents
        self._agent_cache: dict[str, type] = {}

        # Callbacks
        self.on_agent_created: Callable[[AgentBlueprint], None] | None = None
        self.on_agent_retired: Callable[[AgentBlueprint], None] | None = None

    def _init_db(self):
        """Initialize agent factory database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blueprints (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                description TEXT,
                system_prompt TEXT NOT NULL,
                capabilities TEXT,
                tools TEXT,
                parent_id TEXT,
                generation INTEGER DEFAULT 0,
                tasks_completed INTEGER DEFAULT 0,
                tasks_failed INTEGER DEFAULT 0,
                avg_tokens REAL DEFAULT 0,
                avg_duration REAL DEFAULT 0,
                status TEXT DEFAULT 'active',
                code TEXT,
                created_at TEXT,
                retired_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_seen TEXT,
                handled_by TEXT,
                success_rate REAL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_blueprints_status
            ON blueprints(status)
        """)

        conn.commit()
        conn.close()

    def analyze_needs(self) -> list[AgentNeed]:
        """Analyze task patterns to identify needs for new agents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find patterns with low success or high frequency without dedicated agent
        cursor.execute("""
            SELECT pattern, frequency, success_rate, handled_by
            FROM task_patterns
            WHERE frequency >= ?
            ORDER BY frequency DESC
            LIMIT 20
        """, (self.MIN_PATTERN_FREQUENCY,))

        rows = cursor.fetchall()
        conn.close()

        needs = []
        for pattern, freq, success_rate, handled_by in rows:
            success_rate = success_rate or 0.5

            # Check if there's a gap we can fill
            if success_rate < (1 - self.MIN_SUCCESS_RATE_GAP):
                need = AgentNeed(
                    pattern=pattern,
                    frequency=freq,
                    current_success_rate=success_rate,
                    suggested_role=self._suggest_role(pattern),
                    suggested_capabilities=self._extract_capabilities(pattern),
                    confidence=min(0.9, freq / 10 * (1 - success_rate)),
                )
                needs.append(need)

        return needs

    def _suggest_role(self, pattern: str) -> AgentRole:
        """Suggest an agent role based on task pattern."""
        pattern_lower = pattern.lower()

        if any(w in pattern_lower for w in ["analyze", "review", "examine", "inspect"]):
            return AgentRole.ANALYZER
        elif any(w in pattern_lower for w in ["validate", "check", "verify", "test"]):
            return AgentRole.VALIDATOR
        elif any(w in pattern_lower for w in ["convert", "transform", "translate", "format"]):
            return AgentRole.TRANSFORMER
        elif any(w in pattern_lower for w in ["expert", "specialist", "specific"]):
            return AgentRole.SPECIALIST
        else:
            return AgentRole.EXECUTOR

    def _extract_capabilities(self, pattern: str) -> list[str]:
        """Extract capabilities from a task pattern."""
        # Simple keyword extraction
        keywords = []
        for word in pattern.lower().split():
            if len(word) > 4 and word.isalpha():
                keywords.append(word)
        return keywords[:5]

    def create_agent(
        self,
        name: str,
        role: AgentRole,
        description: str,
        capabilities: list[str],
        parent_id: str | None = None,
    ) -> AgentBlueprint:
        """
        Create a new agent type dynamically.

        This generates actual Python code for a new agent class.
        """
        agent_id = f"agent_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        # Generate system prompt based on role and capabilities
        system_prompt = self._generate_system_prompt(role, description, capabilities)

        # Determine generation
        generation = 0
        if parent_id:
            parent = self.get_blueprint(parent_id)
            if parent:
                generation = parent.generation + 1

        # Create blueprint
        blueprint = AgentBlueprint(
            id=agent_id,
            name=name,
            role=role,
            description=description,
            system_prompt=system_prompt,
            capabilities=capabilities,
            tools=["chat", "code_execution"],
            parent_id=parent_id,
            generation=generation,
            status="testing",
        )

        # Generate code
        code = self._generate_agent_code(blueprint)

        # Store blueprint
        self._store_blueprint(blueprint, code)

        if self.on_agent_created:
            self.on_agent_created(blueprint)

        return blueprint

    def _generate_system_prompt(
        self,
        role: AgentRole,
        description: str,
        capabilities: list[str],
    ) -> str:
        """Generate a system prompt for the agent."""
        role_intros = {
            AgentRole.EXECUTOR: "You are an execution agent that completes tasks efficiently.",
            AgentRole.ANALYZER: "You are an analysis agent that examines and understands complex information.",
            AgentRole.VALIDATOR: "You are a validation agent that checks correctness and quality.",
            AgentRole.TRANSFORMER: "You are a transformation agent that converts between formats.",
            AgentRole.SPECIALIST: "You are a specialist agent with deep domain expertise.",
        }

        prompt = f"""{role_intros.get(role, "You are an AI agent.")}

{description}

Your capabilities include:
{chr(10).join(f"- {cap}" for cap in capabilities)}

Guidelines:
- Be precise and thorough
- Explain your reasoning
- Ask for clarification if needed
- Admit uncertainty when appropriate
"""
        return prompt

    def _generate_agent_code(self, blueprint: AgentBlueprint) -> str:
        """Generate Python code for the agent."""
        template = AGENT_TEMPLATES.get(blueprint.role, AGENT_TEMPLATES[AgentRole.EXECUTOR])

        class_name = "".join(word.title() for word in blueprint.name.split("_")) + "Agent"

        code = template.format(
            class_name=class_name,
            description=blueprint.description,
            capabilities=", ".join(blueprint.capabilities),
            generation=blueprint.generation,
            system_prompt=blueprint.system_prompt.replace('"""', '\\"\\"\\"'),
            specialty=blueprint.capabilities[0] if blueprint.capabilities else "general",
        )

        return code

    def _store_blueprint(self, blueprint: AgentBlueprint, code: str):
        """Store blueprint in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO blueprints
            (id, name, role, description, system_prompt, capabilities, tools,
             parent_id, generation, tasks_completed, tasks_failed, avg_tokens,
             avg_duration, status, code, created_at, retired_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            blueprint.id,
            blueprint.name,
            blueprint.role.value,
            blueprint.description,
            blueprint.system_prompt,
            json.dumps(blueprint.capabilities),
            json.dumps(blueprint.tools),
            blueprint.parent_id,
            blueprint.generation,
            blueprint.tasks_completed,
            blueprint.tasks_failed,
            blueprint.avg_tokens,
            blueprint.avg_duration,
            blueprint.status,
            code,
            blueprint.created_at.isoformat(),
            blueprint.retired_at.isoformat() if blueprint.retired_at else None,
        ))

        conn.commit()
        conn.close()

    def get_blueprint(self, agent_id: str) -> AgentBlueprint | None:
        """Get a blueprint by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM blueprints WHERE id = ?", (agent_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_blueprint(row)

    def _row_to_blueprint(self, row) -> AgentBlueprint:
        """Convert database row to blueprint."""
        return AgentBlueprint(
            id=row[0],
            name=row[1],
            role=AgentRole(row[2]),
            description=row[3],
            system_prompt=row[4],
            capabilities=json.loads(row[5]) if row[5] else [],
            tools=json.loads(row[6]) if row[6] else [],
            parent_id=row[7],
            generation=row[8],
            tasks_completed=row[9],
            tasks_failed=row[10],
            avg_tokens=row[11],
            avg_duration=row[12],
            status=row[13],
            created_at=datetime.fromisoformat(row[15]) if row[15] else datetime.now(),
            retired_at=datetime.fromisoformat(row[16]) if row[16] else None,
        )

    def load_agent(self, agent_id: str) -> type | None:
        """Load a dynamic agent class by ID."""
        if agent_id in self._agent_cache:
            return self._agent_cache[agent_id]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT code FROM blueprints WHERE id = ? AND status = 'active'", (agent_id,))
        row = cursor.fetchone()
        conn.close()

        if not row or not row[0]:
            return None

        code = row[0]

        # Execute the code to get the class
        try:
            # Create a namespace with required imports
            namespace = {
                "BaseAgent": self._get_base_agent_class(),
            }
            exec(code, namespace)

            # Find the agent class in namespace
            for name, obj in namespace.items():
                if isinstance(obj, type) and name.endswith("Agent") and name != "BaseAgent":
                    self._agent_cache[agent_id] = obj
                    return obj
        except Exception as e:
            print(f"Failed to load agent {agent_id}: {e}")

        return None

    def _get_base_agent_class(self):
        """Get the BaseAgent class for dynamic agents."""
        from agents.base import BaseAgent
        return BaseAgent

    def activate_agent(self, agent_id: str) -> bool:
        """Activate a testing agent for production use."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE blueprints SET status = 'active' WHERE id = ? AND status = 'testing'",
            (agent_id,)
        )

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def retire_agent(self, agent_id: str, reason: str = "") -> bool:
        """Retire an underperforming agent."""
        blueprint = self.get_blueprint(agent_id)
        if not blueprint:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE blueprints SET status = 'retired', retired_at = ? WHERE id = ?",
            (datetime.now().isoformat(), agent_id)
        )

        conn.commit()
        conn.close()

        # Remove from cache
        self._agent_cache.pop(agent_id, None)

        if self.on_agent_retired:
            self.on_agent_retired(blueprint)

        return True

    def evolve_agent(self, agent_id: str, mutation: str = "enhance") -> AgentBlueprint | None:
        """
        Evolve an agent by creating a mutated variant.

        Mutations:
        - enhance: Add more capabilities
        - simplify: Focus on core function
        - specialize: Narrow focus for better performance
        - hybridize: Combine with another agent's traits
        """
        parent = self.get_blueprint(agent_id)
        if not parent:
            return None

        # Apply mutation
        new_capabilities = parent.capabilities.copy()
        new_description = parent.description

        if mutation == "enhance":
            new_capabilities.append("advanced_reasoning")
            new_description = f"Enhanced {parent.description}"
        elif mutation == "simplify":
            new_capabilities = new_capabilities[:3]
            new_description = f"Simplified {parent.description}"
        elif mutation == "specialize":
            if new_capabilities:
                new_capabilities = [new_capabilities[0], f"expert_{new_capabilities[0]}"]
            new_description = f"Specialized {parent.description}"

        # Create evolved agent
        child = self.create_agent(
            name=f"{parent.name}_v{parent.generation + 1}",
            role=parent.role,
            description=new_description,
            capabilities=new_capabilities,
            parent_id=parent.id,
        )

        return child

    def record_task(self, agent_id: str, success: bool, tokens: int, duration: float):
        """Record task completion for an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if success:
            cursor.execute("""
                UPDATE blueprints
                SET tasks_completed = tasks_completed + 1,
                    avg_tokens = (avg_tokens * tasks_completed + ?) / (tasks_completed + 1),
                    avg_duration = (avg_duration * tasks_completed + ?) / (tasks_completed + 1)
                WHERE id = ?
            """, (tokens, duration, agent_id))
        else:
            cursor.execute(
                "UPDATE blueprints SET tasks_failed = tasks_failed + 1 WHERE id = ?",
                (agent_id,)
            )

        conn.commit()
        conn.close()

        # Check if agent should be retired
        self._maybe_retire(agent_id)

    def _maybe_retire(self, agent_id: str):
        """Check if an agent should be retired due to poor performance."""
        blueprint = self.get_blueprint(agent_id)
        if not blueprint:
            return

        total = blueprint.tasks_completed + blueprint.tasks_failed
        if total < 10:
            return  # Not enough data

        success_rate = blueprint.tasks_completed / total
        if success_rate < self.RETIREMENT_THRESHOLD:
            self.retire_agent(agent_id, f"Low success rate: {success_rate:.0%}")

    def get_active_agents(self) -> list[AgentBlueprint]:
        """Get all active agent blueprints."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM blueprints WHERE status = 'active'")
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_blueprint(row) for row in rows]

    def get_stats(self) -> dict:
        """Get agent factory statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM blueprints WHERE status = 'active'")
        active = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM blueprints WHERE status = 'testing'")
        testing = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM blueprints WHERE status = 'retired'")
        retired = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(generation) FROM blueprints")
        max_gen = cursor.fetchone()[0] or 0

        cursor.execute("SELECT SUM(tasks_completed), SUM(tasks_failed) FROM blueprints")
        row = cursor.fetchone()
        total_completed = row[0] or 0
        total_failed = row[1] or 0

        conn.close()

        return {
            "active_agents": active,
            "testing_agents": testing,
            "retired_agents": retired,
            "max_generation": max_gen,
            "total_tasks": total_completed + total_failed,
            "overall_success_rate": total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0,
        }

    def format_blueprint(self, blueprint: AgentBlueprint) -> str:
        """Format a blueprint for display."""
        success_rate = 0
        total = blueprint.tasks_completed + blueprint.tasks_failed
        if total > 0:
            success_rate = blueprint.tasks_completed / total

        return f"""## Agent: {blueprint.name}
Role: {blueprint.role.value}
Generation: {blueprint.generation}
Status: {blueprint.status}

Description: {blueprint.description}

Capabilities: {', '.join(blueprint.capabilities)}

Performance:
- Tasks: {blueprint.tasks_completed} completed, {blueprint.tasks_failed} failed
- Success Rate: {success_rate:.1%}
- Avg Tokens: {blueprint.avg_tokens:.0f}
- Avg Duration: {blueprint.avg_duration:.1f}s
"""


# Global instance
_factory: AgentFactory | None = None


def get_agent_factory() -> AgentFactory:
    """Get or create the global agent factory."""
    global _factory
    if _factory is None:
        _factory = AgentFactory()
    return _factory
