"""
Function calling prompt templates for native tool use
"""

import json
import uuid
from typing import List, Optional, Tuple, Dict, Any

from ...models.agent_schemas import ToolDefinition, ToolCall


class FunctionCallingPrompts:
    """Prompt templates and parsing for function-calling agents"""

    @staticmethod
    def get_system_prompt(custom_instructions: Optional[str] = None) -> str:
        """
        Generate the system prompt for function-calling agents.
        Tools are provided separately through the API's native tool mechanism.

        Args:
            custom_instructions: Optional additional instructions

        Returns:
            System prompt string
        """
        custom_text = f"\n\n{custom_instructions}" if custom_instructions else ""

        return f"""You are a helpful AI assistant that can use tools to accomplish tasks.

When you need to gather information or perform actions, use the available tools. You can call multiple tools in sequence if needed.

After gathering sufficient information through tools, provide a clear and helpful response to the user.

## Guidelines:

- Use tools when you need current information or to perform actions
- Think through what information you need before using tools
- If a tool call fails, consider alternative approaches
- Provide accurate, helpful responses based on tool results
- Do not make up information - verify facts using tools when uncertain{custom_text}"""

    @staticmethod
    def parse_tool_calls_openai(response: Dict[str, Any]) -> List[ToolCall]:
        """
        Parse tool calls from OpenAI-format response.

        Args:
            response: The API response containing tool_calls

        Returns:
            List of ToolCall objects
        """
        tool_calls = []

        # Handle OpenAI response format
        choices = response.get("choices", [])
        if not choices:
            return tool_calls

        message = choices[0].get("message", {})
        raw_tool_calls = message.get("tool_calls", [])

        for tc in raw_tool_calls:
            try:
                call_id = tc.get("id", str(uuid.uuid4()))
                function = tc.get("function", {})
                name = function.get("name", "")
                arguments_str = function.get("arguments", "{}")

                # Parse arguments
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments_str}

                tool_calls.append(ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments
                ))
            except Exception:
                continue

        return tool_calls

    @staticmethod
    def parse_tool_calls_anthropic(response: Dict[str, Any]) -> List[ToolCall]:
        """
        Parse tool calls from Anthropic-format response.

        Args:
            response: The API response containing tool_use blocks

        Returns:
            List of ToolCall objects
        """
        tool_calls = []

        content = response.get("content", [])

        for block in content:
            if block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", str(uuid.uuid4())),
                    name=block.get("name", ""),
                    arguments=block.get("input", {})
                ))

        return tool_calls

    @staticmethod
    def format_tool_result_openai(call_id: str, result: str) -> Dict[str, Any]:
        """
        Format a tool result for OpenAI API.

        Args:
            call_id: The tool call ID
            result: The result string

        Returns:
            Message dict for the API
        """
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result
        }

    @staticmethod
    def format_tool_result_anthropic(call_id: str, result: str, is_error: bool = False) -> Dict[str, Any]:
        """
        Format a tool result for Anthropic API.

        Args:
            call_id: The tool call ID
            result: The result string
            is_error: Whether this is an error result

        Returns:
            Content block for the API
        """
        return {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": result,
            "is_error": is_error
        }

    @staticmethod
    def has_tool_calls_openai(response: Dict[str, Any]) -> bool:
        """Check if OpenAI response contains tool calls"""
        choices = response.get("choices", [])
        if not choices:
            return False
        message = choices[0].get("message", {})
        return bool(message.get("tool_calls"))

    @staticmethod
    def has_tool_calls_anthropic(response: Dict[str, Any]) -> bool:
        """Check if Anthropic response contains tool calls"""
        content = response.get("content", [])
        return any(block.get("type") == "tool_use" for block in content)

    @staticmethod
    def get_text_content_openai(response: Dict[str, Any]) -> Optional[str]:
        """Extract text content from OpenAI response"""
        choices = response.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        return message.get("content")

    @staticmethod
    def get_text_content_anthropic(response: Dict[str, Any]) -> Optional[str]:
        """Extract text content from Anthropic response"""
        content = response.get("content", [])
        text_blocks = [block.get("text", "") for block in content if block.get("type") == "text"]
        return "\n".join(text_blocks) if text_blocks else None

    @staticmethod
    def is_stop_reason_openai(response: Dict[str, Any]) -> bool:
        """Check if OpenAI response indicates completion (not tool calls)"""
        choices = response.get("choices", [])
        if not choices:
            return True
        finish_reason = choices[0].get("finish_reason", "")
        return finish_reason == "stop"

    @staticmethod
    def is_stop_reason_anthropic(response: Dict[str, Any]) -> bool:
        """Check if Anthropic response indicates completion (not tool calls)"""
        stop_reason = response.get("stop_reason", "")
        return stop_reason == "end_turn"
