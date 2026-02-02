"""
Stdio transport for MCP - communicates with MCP servers via stdin/stdout
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional, List

from ...utils.logging import setup_logging

logger = setup_logging()


class StdioTransport:
    """
    Transport for MCP servers that communicate via stdin/stdout.
    Spawns a subprocess and communicates using JSON-RPC over stdio.
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 30
    ):
        """
        Initialize stdio transport.

        Args:
            command: Command to run (e.g., "npx", "python")
            args: Arguments for the command
            env: Additional environment variables
            timeout_seconds: Timeout for operations
        """
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout_seconds = timeout_seconds

        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._connected = False

    @property
    def connected(self) -> bool:
        """Check if connected to the server"""
        return self._connected and self._process is not None

    async def connect(self) -> bool:
        """
        Start the MCP server process and establish communication.

        Returns:
            True if connection successful
        """
        try:
            # Merge environment
            full_env = os.environ.copy()
            full_env.update(self.env)

            # Build command
            cmd = [self.command] + self.args

            logger.info(
                "Starting MCP server process",
                command=self.command,
                args=self.args
            )

            # Start process
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env
            )

            # Start reader task
            self._reader_task = asyncio.create_task(self._read_responses())

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
                logger.info("MCP stdio connection established")

                # Send initialized notification
                await self._send_notification("notifications/initialized", {})

                return True
            else:
                logger.error("MCP initialization failed")
                await self.disconnect()
                return False

        except Exception as e:
            logger.error("Failed to connect via stdio", error=str(e))
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect and stop the server process"""
        self._connected = False

        # Cancel reader task
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        # Terminate process
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
            except Exception:
                pass
            self._process = None

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(Exception("Transport disconnected"))
        self._pending_requests.clear()

        logger.info("MCP stdio transport disconnected")

    async def _read_responses(self) -> None:
        """Background task to read responses from stdout"""
        try:
            while self._process and self._process.stdout:
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    message = json.loads(line.decode())
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from MCP server", line=line.decode()[:100])

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error reading MCP responses", error=str(e))

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle a message from the server"""
        # Check if it's a response to a request
        if "id" in message:
            request_id = message["id"]
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if "error" in message:
                    future.set_exception(Exception(message["error"].get("message", "Unknown error")))
                else:
                    future.set_result(message.get("result"))
        else:
            # It's a notification - log but don't process
            method = message.get("method", "unknown")
            logger.debug("Received MCP notification", method=method)

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a request and wait for response"""
        if not self._process or not self._process.stdin:
            raise Exception("Not connected")

        self._request_id += 1
        request_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future

        try:
            # Send request
            data = json.dumps(request) + "\n"
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()

            # Wait for response
            result = await asyncio.wait_for(future, timeout=self.timeout_seconds)
            return result

        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise Exception(f"Request timed out: {method}")

    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a notification (no response expected)"""
        if not self._process or not self._process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        data = json.dumps(notification) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

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
