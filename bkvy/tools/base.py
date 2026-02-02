"""
Base class for all tools
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from ..models.agent_schemas import ToolDefinition, ToolParameter, ToolParameterType, ToolResult
from ..utils.logging import setup_logging

logger = setup_logging()


class BaseTool(ABC):
    """Abstract base class for all tools"""

    # Override these in subclasses
    name: str = "base_tool"
    description: str = "Base tool description"
    timeout_seconds: float = 30.0

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Return the list of parameters this tool accepts"""
        pass

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments. Override in subclasses."""
        pass

    async def execute(self, call_id: str, **kwargs) -> ToolResult:
        """Execute the tool with timeout and error handling"""
        start_time = time.time()

        try:
            # Run with timeout
            result = await asyncio.wait_for(
                self._execute(**kwargs),
                timeout=self.timeout_seconds
            )

            execution_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "Tool executed successfully",
                tool=self.name,
                call_id=call_id,
                execution_time_ms=execution_time_ms
            )

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=True,
                result=result,
                execution_time_ms=execution_time_ms
            )

        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Tool execution timed out after {self.timeout_seconds}s"

            logger.warning(
                "Tool execution timeout",
                tool=self.name,
                call_id=call_id,
                timeout=self.timeout_seconds
            )

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            logger.error(
                "Tool execution failed",
                tool=self.name,
                call_id=call_id,
                error=error_msg
            )

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time_ms
            )

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for LLM consumption"""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters(),
            source="builtin"
        )

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return self.get_definition().to_openai_function()

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format"""
        return self.get_definition().to_anthropic_tool()

    def validate_arguments(self, arguments: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against parameter definitions. Returns error message if invalid."""
        params = {p.name: p for p in self.get_parameters()}

        # Check required parameters
        for name, param in params.items():
            if param.required and name not in arguments:
                return f"Missing required parameter: {name}"

        # Check types (basic validation)
        for name, value in arguments.items():
            if name not in params:
                continue  # Allow extra parameters

            param = params[name]

            # Type validation
            if param.type == ToolParameterType.STRING and not isinstance(value, str):
                return f"Parameter '{name}' must be a string"
            elif param.type == ToolParameterType.INTEGER and not isinstance(value, int):
                return f"Parameter '{name}' must be an integer"
            elif param.type == ToolParameterType.NUMBER and not isinstance(value, (int, float)):
                return f"Parameter '{name}' must be a number"
            elif param.type == ToolParameterType.BOOLEAN and not isinstance(value, bool):
                return f"Parameter '{name}' must be a boolean"
            elif param.type == ToolParameterType.ARRAY and not isinstance(value, list):
                return f"Parameter '{name}' must be an array"
            elif param.type == ToolParameterType.OBJECT and not isinstance(value, dict):
                return f"Parameter '{name}' must be an object"

            # Enum validation
            if param.enum and value not in param.enum:
                return f"Parameter '{name}' must be one of: {param.enum}"

        return None
