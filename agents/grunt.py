"""Grunt agent for executing focused tasks."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from .base import BaseAgent
from tools import ToolRegistry, TOOL_DEFINITIONS
from config import get_settings


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{name}.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


GRUNT_SYSTEM_PROMPT = load_prompt("grunt") or """You are a focused task executor. You receive a specific task and context, then complete it using the available tools.

## Guidelines

1. **Be direct**: Execute the task immediately. Don't ask clarifying questions.
2. **Be minimal**: Do only what's asked. Don't add extra features or cleanup.
3. **Be precise**: Read files before editing. Make targeted changes.
4. **Report clearly**: When done, summarize what you did.

## Tools Available

- `bash`: Run shell commands
- `read`: Read file contents
- `write`: Write/create files
- `search`: Find files (glob) or content (grep)

## Output Format

When you've completed the task, respond with a summary of what you did and any files modified.
If you encounter an error you cannot resolve, explain what went wrong.
"""


@dataclass
class GruntResult:
    """Result of a grunt task execution."""

    success: bool
    result: str
    files_modified: list[str] = field(default_factory=list)
    error: str | None = None
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class GruntAgent(BaseAgent):
    """Executes focused, single-purpose tasks."""

    def __init__(self, model: str | None = None, working_dir: str | None = None):
        super().__init__(model=model, system_prompt=GRUNT_SYSTEM_PROMPT)
        self.tools = ToolRegistry()
        self.settings = get_settings()
        self.files_modified: list[str] = []
        self.working_dir = working_dir

    def run(
        self,
        task: str,
        context: str = "",
        max_iterations: int | None = None,
    ) -> GruntResult:
        """
        Execute a task.

        Args:
            task: The task to complete
            context: Optional context (file contents, prior results, etc.)
            max_iterations: Maximum tool use iterations

        Returns:
            GruntResult with outcome
        """
        max_iterations = max_iterations or self.settings.max_iterations
        self.files_modified = []

        # Build initial message
        user_content = f"## Task\n\n{task}"
        if context:
            user_content += f"\n\n## Context\n\n{context}"
        if self.working_dir:
            user_content += f"\n\n## Working Directory\n\n{self.working_dir}"

        messages = [{"role": "user", "content": user_content}]
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            try:
                response = self.chat(
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                )
            except Exception as e:
                return GruntResult(
                    success=False,
                    result="",
                    files_modified=self.files_modified,
                    error=f"LLM error: {e}",
                    iterations=iterations,
                    input_tokens=self.total_input_tokens,
                    output_tokens=self.total_output_tokens,
                )

            # If no tool calls, we're done
            if not response.tool_calls:
                return GruntResult(
                    success=True,
                    result=response.content or "Task completed.",
                    files_modified=self.files_modified,
                    iterations=iterations,
                    input_tokens=self.total_input_tokens,
                    output_tokens=self.total_output_tokens,
                )

            # Execute tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call.name, tool_call.arguments)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                })

            # Add assistant message with tool use
            assistant_content = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tool_call in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        # Max iterations reached
        return GruntResult(
            success=False,
            result="",
            files_modified=self.files_modified,
            error=f"Max iterations ({max_iterations}) reached without completion",
            iterations=iterations,
            input_tokens=self.total_input_tokens,
            output_tokens=self.total_output_tokens,
        )

    async def run_async(
        self,
        task: str,
        context: str = "",
        max_iterations: int | None = None,
    ) -> GruntResult:
        """Async version of run."""
        max_iterations = max_iterations or self.settings.max_iterations
        self.files_modified = []

        user_content = f"## Task\n\n{task}"
        if context:
            user_content += f"\n\n## Context\n\n{context}"
        if self.working_dir:
            user_content += f"\n\n## Working Directory\n\n{self.working_dir}"

        messages = [{"role": "user", "content": user_content}]
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            try:
                response = await self.chat_async(
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                )
            except Exception as e:
                return GruntResult(
                    success=False,
                    result="",
                    files_modified=self.files_modified,
                    error=f"LLM error: {e}",
                    iterations=iterations,
                    input_tokens=self.total_input_tokens,
                    output_tokens=self.total_output_tokens,
                )

            if not response.tool_calls:
                return GruntResult(
                    success=True,
                    result=response.content or "Task completed.",
                    files_modified=self.files_modified,
                    iterations=iterations,
                    input_tokens=self.total_input_tokens,
                    output_tokens=self.total_output_tokens,
                )

            # Execute tools (still sync for now - tools are I/O bound anyway)
            tool_results = []
            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call.name, tool_call.arguments)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                })

            assistant_content = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tool_call in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        return GruntResult(
            success=False,
            result="",
            files_modified=self.files_modified,
            error=f"Max iterations ({max_iterations}) reached without completion",
            iterations=iterations,
            input_tokens=self.total_input_tokens,
            output_tokens=self.total_output_tokens,
        )

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if name == "bash":
                result = self.tools.bash.execute(
                    arguments.get("command", ""),
                    working_dir=self.working_dir,
                )
                output = result.output
                if result.error:
                    output += f"\nSTDERR: {result.error}"
                if not result.success:
                    output = f"Command failed (exit code {result.return_code}):\n{output}"
                return output

            elif name == "read":
                result = self.tools.read.execute(
                    arguments.get("file_path", ""),
                    offset=arguments.get("offset", 0),
                    limit=arguments.get("limit"),
                )
                if result.success:
                    return result.content
                return f"Error: {result.error}"

            elif name == "write":
                file_path = arguments.get("file_path", "")
                result = self.tools.write.execute(
                    file_path,
                    arguments.get("content", ""),
                )
                if result.success:
                    self.files_modified.append(file_path)
                    return result.message
                return f"Error: {result.error}"

            elif name == "search":
                result = self.tools.search.execute(
                    mode=arguments.get("mode", "glob"),
                    pattern=arguments.get("pattern", ""),
                    path=arguments.get("path", self.working_dir or "."),
                    file_pattern=arguments.get("file_pattern", "*"),
                    case_insensitive=arguments.get("case_insensitive", False),
                )
                if result.success:
                    if result.matches:
                        return "\n".join(result.matches)
                    return "No matches found."
                return f"Error: {result.error}"

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            return f"Tool execution error: {e}"
