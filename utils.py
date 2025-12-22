"""Utility functions for Swarm."""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.resolve()


def get_cwd() -> Path:
    """Get the current working directory."""
    return Path.cwd().resolve()


def relative_path(path: str | Path, base: str | Path | None = None) -> str:
    """
    Get relative path from base directory.

    Args:
        path: Path to make relative
        base: Base directory (defaults to cwd)

    Returns:
        Relative path string
    """
    path = Path(path).resolve()
    base = Path(base or get_cwd()).resolve()

    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to max length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count.

    Args:
        text: Text to count

    Returns:
        Estimated token count (roughly 4 chars per token)
    """
    return len(text) // 4


def count_code_lines(file_path: str | Path) -> int:
    """
    Count non-empty, non-comment lines in a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Number of non-empty, non-comment lines
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        OSError: If there's an error reading the file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped = line.strip()
            # Skip empty lines and comment lines
            if stripped and not stripped.startswith('#'):
                count += 1
    
    return count