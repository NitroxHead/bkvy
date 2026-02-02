"""
MCP manager - manages MCP server connections and lifecycle
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from .client import MCPClient
from ..models.agent_schemas import (
    MCPServerConfig, MCPServerStatus, MCPTransportType, ToolResult
)
from ..tools.registry import ToolRegistry
from ..utils.logging import setup_logging

logger = setup_logging()

# Global manager instance
_mcp_manager: Optional["MCPManager"] = None


class MCPManager:
    """
    Manages MCP server connections and provides tool routing.
    Singleton pattern matching existing bkvy managers.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        config_path: Optional[str] = None
    ):
        """
        Initialize MCP manager.

        Args:
            tool_registry: Tool registry to register discovered tools
            config_path: Path to mcp_servers.json config file
        """
        self.tool_registry = tool_registry
        self.config_path = config_path or "config/mcp_servers.json"

        self._configs: Dict[str, MCPServerConfig] = {}
        self._clients: Dict[str, MCPClient] = {}
        self._connection_times: Dict[str, datetime] = {}
        self._errors: Dict[str, str] = {}

        # Set self as the MCP manager in tool registry
        self.tool_registry.set_mcp_manager(self)

    async def load_configs(self) -> None:
        """Load MCP server configurations from config file"""
        config_file = Path(self.config_path)

        if not config_file.exists():
            logger.info("MCP config file not found, no servers configured", path=self.config_path)
            return

        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            servers = config.get("servers", {})
            for name, server_data in servers.items():
                try:
                    # Parse transport type
                    transport = MCPTransportType(server_data.get("transport", "stdio"))

                    server_config = MCPServerConfig(
                        name=name,
                        transport=transport,
                        command=server_data.get("command"),
                        args=server_data.get("args"),
                        env=server_data.get("env"),
                        url=server_data.get("url"),
                        headers=server_data.get("headers"),
                        auto_start=server_data.get("auto_start", False),
                        timeout_seconds=server_data.get("timeout_seconds", 30)
                    )

                    self._configs[name] = server_config
                    logger.info("Loaded MCP server config", name=name, transport=transport.value)

                except Exception as e:
                    logger.error("Failed to load MCP config", name=name, error=str(e))

            logger.info("Loaded MCP server configs", count=len(self._configs))

        except Exception as e:
            logger.error("Failed to load MCP config file", path=self.config_path, error=str(e))

    async def auto_start_servers(self) -> None:
        """Start all servers configured with auto_start=True"""
        for name, config in self._configs.items():
            if config.auto_start:
                logger.info("Auto-starting MCP server", name=name)
                await self.connect(name)

    def add_config(self, config: MCPServerConfig) -> None:
        """Add or update a server configuration"""
        self._configs[config.name] = config
        logger.info("Added MCP server config", name=config.name)

    def remove_config(self, name: str) -> bool:
        """Remove a server configuration"""
        if name in self._configs:
            del self._configs[name]
            logger.info("Removed MCP server config", name=name)
            return True
        return False

    def get_config(self, name: str) -> Optional[MCPServerConfig]:
        """Get a server configuration by name"""
        return self._configs.get(name)

    def list_configs(self) -> List[str]:
        """List all configured server names"""
        return list(self._configs.keys())

    async def connect(self, name: str, config: Optional[MCPServerConfig] = None) -> bool:
        """
        Connect to an MCP server.

        Args:
            name: Server name
            config: Optional config override (for dynamic servers)

        Returns:
            True if connection successful
        """
        # Get or use provided config
        server_config = config or self._configs.get(name)

        if not server_config:
            self._errors[name] = "No configuration found"
            logger.error("MCP server config not found", name=name)
            return False

        # Disconnect if already connected
        if name in self._clients:
            await self.disconnect(name)

        try:
            # Create client
            client = MCPClient(server_config)

            # Connect
            success = await client.connect()

            if success:
                self._clients[name] = client
                self._connection_times[name] = datetime.utcnow()
                self._errors.pop(name, None)

                # Register discovered tools with tool registry
                for tool_def in client.tools:
                    self.tool_registry.register_mcp_tool(name, tool_def)

                logger.info(
                    "MCP server connected",
                    name=name,
                    tools=client.get_tool_names()
                )
                return True
            else:
                self._errors[name] = "Connection failed"
                return False

        except Exception as e:
            self._errors[name] = str(e)
            logger.error("MCP connect error", name=name, error=str(e))
            return False

    async def disconnect(self, name: str) -> bool:
        """
        Disconnect from an MCP server.

        Args:
            name: Server name

        Returns:
            True if disconnected successfully
        """
        if name not in self._clients:
            return False

        try:
            client = self._clients[name]
            await client.disconnect()

            # Unregister tools
            self.tool_registry.unregister_mcp_server(name)

            del self._clients[name]
            self._connection_times.pop(name, None)

            logger.info("MCP server disconnected", name=name)
            return True

        except Exception as e:
            logger.error("MCP disconnect error", name=name, error=str(e))
            return False

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers"""
        names = list(self._clients.keys())
        for name in names:
            await self.disconnect(name)

        logger.info("All MCP servers disconnected")

    def is_connected(self, name: str) -> bool:
        """Check if a server is connected"""
        client = self._clients.get(name)
        return client is not None and client.connected

    def get_status(self, name: str) -> MCPServerStatus:
        """Get status for a specific server"""
        config = self._configs.get(name)
        client = self._clients.get(name)

        if not config:
            return MCPServerStatus(
                name=name,
                connected=False,
                transport=MCPTransportType.STDIO,
                error="Not configured"
            )

        return MCPServerStatus(
            name=name,
            connected=client is not None and client.connected,
            transport=config.transport,
            tools=client.get_tool_names() if client else [],
            connected_at=self._connection_times.get(name),
            error=self._errors.get(name)
        )

    def get_all_statuses(self) -> Dict[str, MCPServerStatus]:
        """Get status for all configured servers"""
        statuses = {}
        for name in self._configs.keys():
            statuses[name] = self.get_status(name)

        # Include connected but unconfigured servers (dynamic)
        for name in self._clients.keys():
            if name not in statuses:
                statuses[name] = self.get_status(name)

        return statuses

    async def execute_tool(
        self,
        call_id: str,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute a tool on an MCP server.

        Args:
            call_id: Tool call ID for tracking
            server_name: Name of the MCP server
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            ToolResult with success/failure
        """
        start_time = time.time()

        client = self._clients.get(server_name)

        if not client:
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                success=False,
                error=f"MCP server not connected: {server_name}"
            )

        if not client.has_tool(tool_name):
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                success=False,
                error=f"Tool not found on server {server_name}: {tool_name}"
            )

        try:
            result = await client.call_tool(tool_name, arguments)

            execution_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "MCP tool executed",
                server=server_name,
                tool=tool_name,
                execution_time_ms=execution_time_ms
            )

            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            logger.error(
                "MCP tool execution failed",
                server=server_name,
                tool=tool_name,
                error=str(e)
            )

            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time_ms
            )

    def get_all_tools(self) -> List[str]:
        """Get names of all available MCP tools"""
        tools = []
        for client in self._clients.values():
            tools.extend(client.get_tool_names())
        return tools


def get_mcp_manager() -> Optional[MCPManager]:
    """Get the global MCP manager singleton"""
    return _mcp_manager


def init_mcp_manager(
    tool_registry: ToolRegistry,
    config_path: Optional[str] = None
) -> MCPManager:
    """Initialize the global MCP manager"""
    global _mcp_manager

    _mcp_manager = MCPManager(
        tool_registry=tool_registry,
        config_path=config_path
    )

    logger.info("MCP manager initialized")
    return _mcp_manager
