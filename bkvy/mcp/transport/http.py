"""
HTTP transport for MCP - communicates with MCP servers via HTTP
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any, Optional, List

from ...utils.logging import setup_logging

logger = setup_logging()


class HTTPTransport:
    """
    Transport for MCP servers that communicate via HTTP.
    Uses HTTP POST for requests and optionally SSE for streaming.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 30
    ):
        """
        Initialize HTTP transport.

        Args:
            url: Base URL of the MCP server
            headers: Additional HTTP headers
            timeout_seconds: Timeout for operations
        """
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds

        self._session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0
        self._connected = False
        self._server_capabilities: Dict[str, Any] = {}

    @property
    def connected(self) -> bool:
        """Check if connected to the server"""
        return self._connected

    async def connect(self) -> bool:
        """
        Initialize connection to the MCP server.

        Returns:
            True if connection successful
        """
        try:
            # Create session
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                headers={
                    "Content-Type": "application/json",
                    **self.headers
                }
            )

            logger.info("Connecting to MCP HTTP server", url=self.url)

            # Send initialize request
            result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True}
                },
                "clientInfo": {
                    "name": "bkvy",
                    "version": "1.0.0"
                }
            })

            if result is not None:
                self._connected = True
                self._server_capabilities = result.get("capabilities", {})
                logger.info(
                    "MCP HTTP connection established",
                    capabilities=list(self._server_capabilities.keys())
                )

                # Send initialized notification
                await self._send_notification("notifications/initialized", {})

                return True
            else:
                logger.error("MCP HTTP initialization failed")
                await self.disconnect()
                return False

        except Exception as e:
            logger.error("Failed to connect via HTTP", error=str(e))
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from the server"""
        self._connected = False

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("MCP HTTP transport disconnected")

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a request and get response"""
        if not self._session:
            raise Exception("Not connected")

        self._request_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params
        }

        try:
            async with self._session.post(self.url, json=request) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                result = await response.json()

                if "error" in result:
                    error = result["error"]
                    raise Exception(error.get("message", "Unknown error"))

                return result.get("result")

        except aiohttp.ClientError as e:
            raise Exception(f"HTTP request failed: {e}")

    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a notification (no response expected)"""
        if not self._session:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        try:
            async with self._session.post(self.url, json=notification) as response:
                # Notifications don't expect a specific response
                pass
        except Exception as e:
            logger.warning("Failed to send notification", method=method, error=str(e))

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server"""
        if not self._connected:
            raise Exception("Not connected")

        result = await self._send_request("tools/list", {})
        return result.get("tools", []) if result else []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server"""
        if not self._connected:
            raise Exception("Not connected")

        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })

        return result
