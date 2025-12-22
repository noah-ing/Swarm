"""Write tool for creating and modifying files."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WriteResult:
    """Result of a file write operation."""

    success: bool
    message: str
    error: str | None = None
    bytes_written: int = 0


class WriteTool:
    """Write content to files."""

    name = "write"
    description = "Write content to a file, creating directories if needed."

    def execute(
        self,
        file_path: str,
        content: str,
        create_dirs: bool = True,
    ) -> WriteResult:
        """
        Write content to a file.

        Args:
            file_path: Path to the file to write
            content: Content to write
            create_dirs: Create parent directories if they don't exist

        Returns:
            WriteResult with status
        """
        path = Path(file_path).expanduser().resolve()

        try:
            # Create parent directories if needed
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            with open(path, "w", encoding="utf-8") as f:
                bytes_written = f.write(content)

            return WriteResult(
                success=True,
                message=f"Successfully wrote {bytes_written} bytes to {path}",
                bytes_written=bytes_written,
            )

        except PermissionError:
            return WriteResult(
                success=False,
                message="",
                error=f"Permission denied: {file_path}",
            )
        except Exception as e:
            return WriteResult(
                success=False,
                message="",
                error=str(e),
            )


# Tool definition for LLM
WRITE_TOOL_DEFINITION = {
    "name": "write",
    "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Creates parent directories as needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    },
}
