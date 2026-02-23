"""
MCP client - handles communication with MCP servers
"""

from typing import Dict, List, Any, Optional, Union

from .transport.stdio import StdioTransport
from .transport.http import HTTPTransport
from ..models.agent_schemas import (
    MCPServerConfig, MCPTransportType, ToolDefinition, ToolParameter, ToolParameterType
)
from ..utils.logging import setup_logging

logger = setup_logging()


class MCPClient:
    """
    Client for communicating with an MCP server.
    Handles connection, tool discovery, and tool execution.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self._transport: Optional[Union[StdioTransport, HTTPTransport]] = None
        self._tools: List[ToolDefinition] = []

    @property
    def name(self) -> str:
        """Get server name"""
        return self.config.name

    @property
    def connected(self) -> bool:
        """Check if connected"""
        return self._transport is not None and self._transport.connected

    @property
    def tools(self) -> List[ToolDefinition]:
        """Get discovered tools"""
        return self._tools.copy()

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        Returns:
            True if connection successful
        """
        try:
            # Create appropriate transport
            if self.config.transport == MCPTransportType.STDIO:
                if not self.config.command:
                    raise ValueError("Stdio transport requires 'command' in config")

                self._transport = StdioTransport(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env,
                    timeout_seconds=self.config.timeout_seconds
                )
            elif self.config.transport == MCPTransportType.HTTP:
                if not self.config.url:
                    raise ValueError("HTTP transport requires 'url' in config")

                self._transport = HTTPTransport(
                    url=self.config.url,
                    headers=self.config.headers,
                    timeout_seconds=self.config.timeout_seconds
                )
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")

            # Connect
            success = await self._transport.connect()

            if success:
                # Discover tools
                await self._discover_tools()
                logger.info(
                    "MCP client connected",
                    server=self.config.name,
                    tools_discovered=len(self._tools)
                )
            else:
                logger.error("MCP client connection failed", server=self.config.name)

            return success

        except Exception as e:
            logger.error("MCP client connect error", server=self.config.name, error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        if self._transport:
            await self._transport.disconnect()
            self._transport = None

        self._tools.clear()
        logger.info("MCP client disconnected", server=self.config.name)

    async def _discover_tools(self) -> None:
        """Discover available tools from the server"""
        if not self._transport:
            return

        try:
            raw_tools = await self._transport.list_tools()

            self._tools = []
            for raw_tool in raw_tools:
                tool_def = self._convert_tool_definition(raw_tool)
                if tool_def:
                    self._tools.append(tool_def)

            logger.info(
                "Discovered MCP tools",
                server=self.config.name,
                tools=[t.name for t in self._tools]
            )

        except Exception as e:
            logger.error("Failed to discover tools", server=self.config.name, error=str(e))

    def _convert_tool_definition(self, raw: Dict[str, Any]) -> Optional[ToolDefinition]:
        """Convert raw MCP tool definition to our format"""
        try:
            name = raw.get("name", "")
            description = raw.get("description", "")
            input_schema = raw.get("inputSchema", {})

            # Parse parameters from JSON Schema
            parameters = []
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            for param_name, param_schema in properties.items():
                param_type = self._map_json_schema_type(param_schema.get("type", "string"))

                parameters.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_schema.get("description", ""),
                    required=param_name in required,
                    enum=param_schema.get("enum"),
                    items=param_schema.get("items")
                ))

            return ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                source="mcp",
                mcp_server=self.config.name
            )

        except Exception as e:
            logger.warning(
                "Failed to convert tool definition",
                server=self.config.name,
                error=str(e)
            )
            return None

    def _map_json_schema_type(self, schema_type: str) -> ToolParameterType:
        """Map JSON Schema type to ToolParameterType"""
        type_map = {
            "string": ToolParameterType.STRING,
            "integer": ToolParameterType.INTEGER,
            "number": ToolParameterType.NUMBER,
            "boolean": ToolParameterType.BOOLEAN,
            "array": ToolParameterType.ARRAY,
            "object": ToolParameterType.OBJECT
        }
        return type_map.get(schema_type, ToolParameterType.STRING)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self._transport or not self._transport.connected:
            raise Exception(f"Not connected to server: {self.config.name}")

        logger.info(
            "Calling MCP tool",
            server=self.config.name,
            tool=name
        )

        result = await self._transport.call_tool(name, arguments)

        # Handle MCP tool result format
        if isinstance(result, dict):
            content = result.get("content", [])
            if content and isinstance(content, list):
                # Extract text content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                if text_parts:
                    return "\n".join(text_parts)

            # Check for error
            if result.get("isError"):
                raise Exception(result.get("content", [{}])[0].get("text", "Unknown error"))

        return result

    def get_tool_names(self) -> List[str]:
        """Get list of discovered tool names"""
        return [tool.name for tool in self._tools]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is available"""
        return any(tool.name == name for tool in self._tools)
