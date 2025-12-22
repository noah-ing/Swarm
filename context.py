"""Context manager for intelligent file context passing to grunts."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from tools import SearchTool, ReadTool


@dataclass
class FileContext:
    """Represents a file with its content."""

    path: str
    content: str
    line_count: int
    relevance: float = 1.0  # 0-1, how relevant to the task


@dataclass
class Context:
    """Manages context for task execution."""

    files: list[FileContext] = field(default_factory=list)
    prior_results: list[str] = field(default_factory=list)
    working_dir: str = field(default_factory=os.getcwd)
    max_context_chars: int = 50000  # ~12k tokens

    def add_file(self, path: str, relevance: float = 1.0) -> bool:
        """
        Add a file to the context.

        Args:
            path: Path to the file
            relevance: How relevant (0-1) the file is

        Returns:
            True if file was added successfully
        """
        read_tool = ReadTool()
        result = read_tool.execute(path)

        if not result.success:
            return False

        self.files.append(FileContext(
            path=path,
            content=result.content,
            line_count=result.line_count,
            relevance=relevance,
        ))
        return True

    def add_files_by_pattern(self, pattern: str, max_files: int = 10) -> int:
        """
        Add files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "src/**/*.py")
            max_files: Maximum number of files to add

        Returns:
            Number of files added
        """
        search_tool = SearchTool()
        result = search_tool.glob(pattern, self.working_dir, max_results=max_files)

        if not result.success:
            return 0

        count = 0
        for path in result.matches:
            if self.add_file(path):
                count += 1

        return count

    def add_files_by_content(
        self,
        search_pattern: str,
        file_pattern: str = "*",
        max_files: int = 5,
    ) -> int:
        """
        Add files containing a pattern.

        Args:
            search_pattern: Regex pattern to search for
            file_pattern: Filter files by pattern
            max_files: Maximum number of files to add

        Returns:
            Number of files added
        """
        search_tool = SearchTool()
        result = search_tool.grep(
            search_pattern,
            self.working_dir,
            file_pattern,
            max_results=max_files * 5,  # Get more results, dedupe later
        )

        if not result.success:
            return 0

        # Extract unique file paths
        seen_paths = set()
        count = 0

        for match in result.matches:
            # Format is "path:line: content"
            path = match.split(":")[0]
            if path not in seen_paths:
                seen_paths.add(path)
                if self.add_file(path, relevance=0.8):
                    count += 1
                    if count >= max_files:
                        break

        return count

    def add_result(self, result: str) -> None:
        """Add a prior result to context."""
        self.prior_results.append(result)

    def build(self, max_chars: int | None = None) -> str:
        """
        Build the context string for a grunt.

        Args:
            max_chars: Maximum characters (defaults to max_context_chars)

        Returns:
            Formatted context string
        """
        max_chars = max_chars or self.max_context_chars
        parts = []
        current_chars = 0

        # Add prior results first (usually smaller, more important)
        if self.prior_results:
            parts.append("## Prior Results\n")
            for i, result in enumerate(self.prior_results, 1):
                result_section = f"\n### Result {i}\n{result}\n"
                if current_chars + len(result_section) > max_chars:
                    break
                parts.append(result_section)
                current_chars += len(result_section)

        # Add files, sorted by relevance
        if self.files:
            sorted_files = sorted(self.files, key=lambda f: f.relevance, reverse=True)
            parts.append("\n## Relevant Files\n")

            for file_ctx in sorted_files:
                file_section = f"\n### {file_ctx.path}\n```\n{file_ctx.content}\n```\n"

                # Truncate if needed
                if current_chars + len(file_section) > max_chars:
                    remaining = max_chars - current_chars - 100
                    if remaining > 500:  # Only include if we can show meaningful content
                        truncated_content = file_ctx.content[:remaining]
                        file_section = f"\n### {file_ctx.path} (truncated)\n```\n{truncated_content}\n...\n```\n"
                        parts.append(file_section)
                    break

                parts.append(file_section)
                current_chars += len(file_section)

        return "".join(parts)

    def clear(self) -> None:
        """Clear all context."""
        self.files = []
        self.prior_results = []


class ContextBuilder:
    """Builder pattern for creating context."""

    def __init__(self, working_dir: str | None = None):
        self.context = Context(working_dir=working_dir or os.getcwd())

    def with_file(self, path: str, relevance: float = 1.0) -> "ContextBuilder":
        """Add a specific file."""
        self.context.add_file(path, relevance)
        return self

    def with_files(self, pattern: str, max_files: int = 10) -> "ContextBuilder":
        """Add files by glob pattern."""
        self.context.add_files_by_pattern(pattern, max_files)
        return self

    def with_search(
        self,
        pattern: str,
        file_pattern: str = "*",
        max_files: int = 5,
    ) -> "ContextBuilder":
        """Add files containing a pattern."""
        self.context.add_files_by_content(pattern, file_pattern, max_files)
        return self

    def with_result(self, result: str) -> "ContextBuilder":
        """Add a prior result."""
        self.context.add_result(result)
        return self

    def build(self, max_chars: int | None = None) -> str:
        """Build the context string."""
        return self.context.build(max_chars)

    def get_context(self) -> Context:
        """Get the underlying Context object."""
        return self.context
