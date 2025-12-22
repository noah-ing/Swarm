"""Search tool for finding files and content."""

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path

from config import get_settings


@dataclass
class SearchResult:
    """Result of a search operation."""

    success: bool
    matches: list[str]
    error: str | None = None
    total_matches: int = 0


class SearchTool:
    """Search for files and content using glob patterns and regex."""

    name = "search"
    description = "Search for files by pattern or content."

    def __init__(self, max_results: int = 100):
        self.max_results = max_results

    def glob(
        self,
        pattern: str,
        path: str = ".",
        max_results: int | None = None,
    ) -> SearchResult:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py")
            path: Directory to search in
            max_results: Maximum number of results

        Returns:
            SearchResult with matching file paths
        """
        max_results = max_results or self.max_results
        base_path = Path(path).expanduser().resolve()

        if not base_path.exists():
            return SearchResult(
                success=False,
                matches=[],
                error=f"Path not found: {path}",
            )

        try:
            matches = []
            for match in base_path.glob(pattern):
                if match.is_file():
                    matches.append(str(match))
                    if len(matches) >= max_results:
                        break

            # Sort by modification time (most recent first)
            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            return SearchResult(
                success=True,
                matches=matches,
                total_matches=len(matches),
            )

        except Exception as e:
            return SearchResult(
                success=False,
                matches=[],
                error=str(e),
            )

    def grep(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        max_results: int | None = None,
        case_insensitive: bool = False,
    ) -> SearchResult:
        """
        Search file contents for a pattern.

        Args:
            pattern: Regex pattern to search for
            path: Directory to search in
            file_pattern: Glob pattern to filter files
            max_results: Maximum number of matches
            case_insensitive: Case-insensitive search

        Returns:
            SearchResult with matching lines
        """
        max_results = max_results or self.max_results
        base_path = Path(path).expanduser().resolve()

        if not base_path.exists():
            return SearchResult(
                success=False,
                matches=[],
                error=f"Path not found: {path}",
            )

        flags = re.IGNORECASE if case_insensitive else 0

        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return SearchResult(
                success=False,
                matches=[],
                error=f"Invalid regex pattern: {e}",
            )

        try:
            matches = []

            # Walk directory tree
            for root, dirs, files in os.walk(base_path):
                # Skip hidden and common ignore directories
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", "venv", ".venv")]

                for filename in files:
                    if not fnmatch.fnmatch(filename, file_pattern):
                        continue

                    file_path = Path(root) / filename

                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            for line_num, line in enumerate(f, 1):
                                if regex.search(line):
                                    match_str = f"{file_path}:{line_num}: {line.rstrip()[:200]}"
                                    matches.append(match_str)

                                    if len(matches) >= max_results:
                                        return SearchResult(
                                            success=True,
                                            matches=matches,
                                            total_matches=len(matches),
                                        )
                    except (IOError, OSError):
                        continue

            return SearchResult(
                success=True,
                matches=matches,
                total_matches=len(matches),
            )

        except Exception as e:
            return SearchResult(
                success=False,
                matches=[],
                error=str(e),
            )

    def execute(
        self,
        mode: str,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        case_insensitive: bool = False,
    ) -> SearchResult:
        """
        Unified search interface.

        Args:
            mode: "glob" for file patterns, "grep" for content search
            pattern: Pattern to search for
            path: Directory to search in
            file_pattern: (grep only) Filter files by pattern
            case_insensitive: (grep only) Case-insensitive search

        Returns:
            SearchResult
        """
        if mode == "glob":
            return self.glob(pattern, path)
        elif mode == "grep":
            return self.grep(pattern, path, file_pattern, case_insensitive=case_insensitive)
        else:
            return SearchResult(
                success=False,
                matches=[],
                error=f"Unknown search mode: {mode}. Use 'glob' or 'grep'.",
            )


# Tool definition for LLM
SEARCH_TOOL_DEFINITION = {
    "name": "search",
    "description": "Search for files or content. Use mode='glob' to find files by pattern (e.g., '**/*.py'). Use mode='grep' to search file contents with regex.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["glob", "grep"],
                "description": "'glob' to find files by pattern, 'grep' to search content",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern (for mode=glob) or regex pattern (for mode=grep)",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Defaults to current directory.",
            },
            "file_pattern": {
                "type": "string",
                "description": "(grep only) Filter files by glob pattern, e.g., '*.py'",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "(grep only) Case-insensitive search",
            },
        },
        "required": ["mode", "pattern"],
    },
}
