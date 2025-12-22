"""Bash tool for executing shell commands."""

import subprocess
from dataclasses import dataclass

from config import get_settings


@dataclass
class BashResult:
    """Result of a bash command execution."""

    success: bool
    output: str
    error: str | None = None
    return_code: int = 0


class BashTool:
    """Execute shell commands with timeout and output limits."""

    name = "bash"
    description = "Execute a shell command and return the output."

    def __init__(self, timeout: int | None = None, max_output: int | None = None):
        settings = get_settings()
        self.timeout = timeout or settings.bash_timeout
        self.max_output = max_output or settings.max_output_length

    def execute(self, command: str, working_dir: str | None = None) -> BashResult:
        """
        Execute a shell command.

        Args:
            command: The shell command to execute
            working_dir: Optional working directory

        Returns:
            BashResult with output and status
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=working_dir,
            )

            output = result.stdout
            error = result.stderr if result.stderr else None

            # Truncate if needed
            if len(output) > self.max_output:
                output = output[: self.max_output] + f"\n... (truncated, {len(output)} total chars)"

            if error and len(error) > self.max_output:
                error = error[: self.max_output] + f"\n... (truncated, {len(error)} total chars)"

            return BashResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                return_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return BashResult(
                success=False,
                output="",
                error=f"Command timed out after {self.timeout} seconds",
                return_code=-1,
            )
        except Exception as e:
            return BashResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
            )


# Tool definition for LLM
BASH_TOOL_DEFINITION = {
    "name": "bash",
    "description": "Execute a shell command. Use for running programs, git operations, file system commands, etc.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
        },
        "required": ["command"],
    },
}
