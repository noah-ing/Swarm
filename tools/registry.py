"""Tool registry for managing available tools."""

from typing import Any

from .bash import BashTool, BASH_TOOL_DEFINITION
from .read import ReadTool, READ_TOOL_DEFINITION
from .write import WriteTool, WRITE_TOOL_DEFINITION
from .search import SearchTool, SEARCH_TOOL_DEFINITION


# All tool definitions for LLM
TOOL_DEFINITIONS = [
    BASH_TOOL_DEFINITION,
    READ_TOOL_DEFINITION,
    WRITE_TOOL_DEFINITION,
    SEARCH_TOOL_DEFINITION,
]


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self.bash = BashTool()
        self.read = ReadTool()
        self.write = WriteTool()
        self.search = SearchTool()

        self._tools = {
            "bash": self.bash,
            "read": self.read,
            "write": self.write,
            "search": self.search,
        }

    def execute(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool result
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        return tool.execute(**kwargs)

    def get_definitions(self) -> list[dict]:
        """Get tool definitions for LLM."""
        return TOOL_DEFINITIONS

    def list_tools(self) -> list[str]:
        """List available tool names."""
        return list(self._tools.keys())
