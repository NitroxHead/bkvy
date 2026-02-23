"""
Tool registry for managing available tools
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .base import BaseTool
from ..models.agent_schemas import ToolDefinition, ToolResult, ToolCall
from ..utils.logging import setup_logging

if TYPE_CHECKING:
    from ..mcp.manager import MCPManager

logger = setup_logging()

# Global registry instance
_tool_registry: Optional["ToolRegistry"] = None


class ToolRegistry:
    """
    Central registry for all tools available to agents.
    Manages built-in tools, MCP tools, and custom tools.
    """

    def __init__(self):
        self._builtin_tools: Dict[str, BaseTool] = {}
        self._mcp_tools: Dict[str, ToolDefinition] = {}  # name -> definition
        self._mcp_server_map: Dict[str, str] = {}  # tool_name -> server_name
        self._custom_tools: Dict[str, BaseTool] = {}
        self._mcp_manager: Optional["MCPManager"] = None

    def set_mcp_manager(self, mcp_manager: "MCPManager"):
        """Set the MCP manager for executing MCP tools"""
        self._mcp_manager = mcp_manager

    # =========================================================================
    # Built-in Tool Management
    # =========================================================================

    def register_builtin(self, tool: BaseTool) -> None:
        """Register a built-in tool"""
        if tool.name in self._builtin_tools:
            logger.warning("Overwriting existing builtin tool", tool=tool.name)

        self._builtin_tools[tool.name] = tool
        logger.info("Registered builtin tool", tool=tool.name)

    def unregister_builtin(self, name: str) -> bool:
        """Unregister a built-in tool"""
        if name in self._builtin_tools:
            del self._builtin_tools[name]
            logger.info("Unregistered builtin tool", tool=name)
            return True
        return False

    # =========================================================================
    # MCP Tool Management
    # =========================================================================

    def register_mcp_tool(self, server_name: str, definition: ToolDefinition) -> None:
        """Register an MCP tool discovered from a server"""
        full_name = f"mcp:{server_name}:{definition.name}"

        # Also track with original name for lookup
        self._mcp_tools[full_name] = definition
        self._mcp_tools[definition.name] = definition
        self._mcp_server_map[full_name] = server_name
        self._mcp_server_map[definition.name] = server_name

        logger.info("Registered MCP tool", tool=full_name, server=server_name)

    def unregister_mcp_server(self, server_name: str) -> int:
        """Unregister all tools from an MCP server"""
        count = 0
        to_remove = []

        for name, sn in self._mcp_server_map.items():
            if sn == server_name:
                to_remove.append(name)

        for name in to_remove:
            if name in self._mcp_tools:
                del self._mcp_tools[name]
            if name in self._mcp_server_map:
                del self._mcp_server_map[name]
            count += 1

        logger.info("Unregistered MCP server tools", server=server_name, count=count)
        return count

    # =========================================================================
    # Custom Tool Management
    # =========================================================================

    def register_custom(self, tool: BaseTool) -> None:
        """Register a custom tool"""
        if tool.name in self._custom_tools:
            logger.warning("Overwriting existing custom tool", tool=tool.name)

        self._custom_tools[tool.name] = tool
        logger.info("Registered custom tool", tool=tool.name)

    def unregister_custom(self, name: str) -> bool:
        """Unregister a custom tool"""
        if name in self._custom_tools:
            del self._custom_tools[name]
            logger.info("Unregistered custom tool", tool=name)
            return True
        return False

    # =========================================================================
    # Tool Lookup
    # =========================================================================

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a built-in or custom tool by name"""
        if name in self._builtin_tools:
            return self._builtin_tools[name]
        if name in self._custom_tools:
            return self._custom_tools[name]
        return None

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name (any source)"""
        # Check built-in
        if name in self._builtin_tools:
            return self._builtin_tools[name].get_definition()

        # Check custom
        if name in self._custom_tools:
            return self._custom_tools[name].get_definition()

        # Check MCP
        if name in self._mcp_tools:
            return self._mcp_tools[name]

        return None

    def list_all(self) -> List[ToolDefinition]:
        """List all available tool definitions"""
        definitions = []

        # Built-in tools
        for tool in self._builtin_tools.values():
            definitions.append(tool.get_definition())

        # Custom tools
        for tool in self._custom_tools.values():
            definitions.append(tool.get_definition())

        # MCP tools (only full names to avoid duplicates)
        seen = set()
        for name, defn in self._mcp_tools.items():
            if name.startswith("mcp:") and defn.name not in seen:
                definitions.append(defn)
                seen.add(defn.name)

        return definitions

    def list_by_names(self, names: List[str]) -> List[ToolDefinition]:
        """Get tool definitions for specific tool names"""
        definitions = []

        for name in names:
            defn = self.get_definition(name)
            if defn:
                definitions.append(defn)
            else:
                logger.warning("Tool not found", tool=name)

        return definitions

    def list_builtin_names(self) -> List[str]:
        """List names of all built-in tools"""
        return list(self._builtin_tools.keys())

    def list_mcp_names(self) -> List[str]:
        """List names of all MCP tools"""
        # Return only the short names
        return [name for name in self._mcp_tools.keys() if not name.startswith("mcp:")]

    def list_custom_names(self) -> List[str]:
        """List names of all custom tools"""
        return list(self._custom_tools.keys())

    # =========================================================================
    # Tool Execution
    # =========================================================================

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call, routing to the correct source"""
        name = call.name

        # Check built-in tools first
        if name in self._builtin_tools:
            tool = self._builtin_tools[name]
            validation_error = tool.validate_arguments(call.arguments)
            if validation_error:
                return ToolResult(
                    call_id=call.id,
                    tool_name=name,
                    success=False,
                    error=validation_error
                )
            return await tool.execute(call.id, **call.arguments)

        # Check custom tools
        if name in self._custom_tools:
            tool = self._custom_tools[name]
            validation_error = tool.validate_arguments(call.arguments)
            if validation_error:
                return ToolResult(
                    call_id=call.id,
                    tool_name=name,
                    success=False,
                    error=validation_error
                )
            return await tool.execute(call.id, **call.arguments)

        # Check MCP tools
        if name in self._mcp_server_map:
            server_name = self._mcp_server_map[name]

            if not self._mcp_manager:
                return ToolResult(
                    call_id=call.id,
                    tool_name=name,
                    success=False,
                    error="MCP manager not configured"
                )

            # Get the actual tool name (remove mcp:server: prefix if present)
            actual_name = name
            if name.startswith("mcp:"):
                parts = name.split(":", 2)
                if len(parts) == 3:
                    actual_name = parts[2]

            return await self._mcp_manager.execute_tool(
                call_id=call.id,
                server_name=server_name,
                tool_name=actual_name,
                arguments=call.arguments
            )

        # Tool not found
        return ToolResult(
            call_id=call.id,
            tool_name=name,
            success=False,
            error=f"Tool not found: {name}"
        )

    # =========================================================================
    # LLM Format Conversion
    # =========================================================================

    def to_openai_tools(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function calling format"""
        if names:
            definitions = self.list_by_names(names)
        else:
            definitions = self.list_all()

        return [defn.to_openai_function() for defn in definitions]

    def to_anthropic_tools(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic tool format"""
        if names:
            definitions = self.list_by_names(names)
        else:
            definitions = self.list_all()

        return [defn.to_anthropic_tool() for defn in definitions]


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry singleton"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def init_tool_registry() -> ToolRegistry:
    """Initialize the tool registry with built-in tools"""
    global _tool_registry
    _tool_registry = ToolRegistry()

    # Register built-in tools
    from .builtin import WebFetchTool, WebSearchTool

    _tool_registry.register_builtin(WebFetchTool())
    _tool_registry.register_builtin(WebSearchTool())

    logger.info(
        "Tool registry initialized",
        builtin_tools=_tool_registry.list_builtin_names()
    )

    return _tool_registry
