"""Read tool for reading file contents."""

import os
from dataclasses import dataclass
from pathlib import Path

from config import get_settings


@dataclass
class ReadResult:
    """Result of a file read operation."""

    success: bool
    content: str
    error: str | None = None
    line_count: int = 0


class ReadTool:
    """Read file contents with line limits."""

    name = "read"
    description = "Read the contents of a file."

    def __init__(self, max_lines: int = 2000, max_chars: int | None = None):
        settings = get_settings()
        self.max_lines = max_lines
        self.max_chars = max_chars or settings.max_output_length

    def execute(
        self,
        file_path: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> ReadResult:
        """
        Read a file's contents.

        Args:
            file_path: Path to the file to read
            offset: Line number to start from (0-indexed)
            limit: Maximum number of lines to read

        Returns:
            ReadResult with file contents
        """
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return ReadResult(
                success=False,
                content="",
                error=f"File not found: {file_path}",
            )

        if not path.is_file():
            return ReadResult(
                success=False,
                content="",
                error=f"Not a file: {file_path}",
            )

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            limit = limit or self.max_lines

            # Apply offset and limit
            selected_lines = lines[offset : offset + limit]

            # Add line numbers
            numbered_lines = []
            for i, line in enumerate(selected_lines, start=offset + 1):
                # Truncate long lines
                if len(line) > 2000:
                    line = line[:2000] + "... (truncated)\n"
                numbered_lines.append(f"{i:6}\t{line.rstrip()}")

            content = "\n".join(numbered_lines)

            # Add truncation notice if needed
            if offset + limit < total_lines:
                content += f"\n\n... ({total_lines - offset - limit} more lines)"

            return ReadResult(
                success=True,
                content=content,
                line_count=total_lines,
            )

        except Exception as e:
            return ReadResult(
                success=False,
                content="",
                error=str(e),
            )


# Tool definition for LLM
READ_TOOL_DEFINITION = {
    "name": "read",
    "description": "Read the contents of a file. Returns line-numbered content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start from (0-indexed). Optional.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read. Optional.",
            },
        },
        "required": ["file_path"],
    },
}
