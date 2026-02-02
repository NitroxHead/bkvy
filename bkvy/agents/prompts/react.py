"""
ReAct (Reasoning + Acting) prompt templates
"""

import re
from typing import List, Optional, Tuple, Dict, Any

from ...models.agent_schemas import ToolDefinition, ToolCall


class ReActPrompts:
    """Prompt templates and parsing for ReAct-style agents"""

    @staticmethod
    def get_system_prompt(tools: List[ToolDefinition], custom_instructions: Optional[str] = None) -> str:
        """
        Generate the ReAct system prompt with available tools.

        Args:
            tools: List of available tool definitions
            custom_instructions: Optional additional instructions

        Returns:
            System prompt string
        """
        tool_descriptions = []
        for tool in tools:
            params = []
            for param in tool.parameters:
                required = "(required)" if param.required else "(optional)"
                params.append(f"    - {param.name}: {param.type.value} {required} - {param.description}")

            param_str = "\n".join(params) if params else "    (no parameters)"
            tool_descriptions.append(f"- {tool.name}: {tool.description}\n  Parameters:\n{param_str}")

        tools_text = "\n\n".join(tool_descriptions) if tool_descriptions else "(No tools available)"

        custom_text = f"\n\nAdditional Instructions:\n{custom_instructions}" if custom_instructions else ""

        return f"""You are a helpful AI assistant that can use tools to accomplish tasks.

You should follow the ReAct pattern:
1. **Thought**: Reason about the current situation and what to do next
2. **Action**: If you need to use a tool, specify the action
3. **Observation**: You will receive the result of your action

When you have gathered enough information to answer the user's question or complete their task, provide a final response.

## Available Tools:

{tools_text}

## Response Format:

When reasoning and using tools, use this format:

Thought: [Your reasoning about what to do]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}

After receiving an observation, continue with another Thought/Action cycle or provide a final answer.

When you have the final answer, respond with:

Thought: [Your final reasoning]
Final Answer: [Your complete response to the user]{custom_text}

## Important Guidelines:

- Always think before acting
- Use tools when needed to gather information
- Be precise with tool parameters
- Provide helpful, accurate final answers
- If a tool fails, consider alternative approaches
- Do not make up information - use tools to verify facts"""

    @staticmethod
    def parse_action(response: str) -> Tuple[Optional[ToolCall], Optional[str], Optional[str]]:
        """
        Parse a ReAct response to extract action, thought, or final answer.

        Args:
            response: The LLM's response text

        Returns:
            Tuple of (tool_call, thought, final_answer)
            - tool_call: ToolCall if an action was specified
            - thought: The thought/reasoning text
            - final_answer: Final answer if present
        """
        import json
        import uuid

        thought = None
        tool_call = None
        final_answer = None

        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Check for final answer
        final_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)
        if final_match:
            final_answer = final_match.group(1).strip()
            return tool_call, thought, final_answer

        # Extract action
        action_match = re.search(r"Action:\s*(\S+)", response, re.IGNORECASE)
        if action_match:
            action_name = action_match.group(1).strip()

            # Extract action input
            input_match = re.search(r"Action Input:\s*(\{.+?\}|\".+?\"|\S+)", response, re.DOTALL | re.IGNORECASE)

            arguments = {}
            if input_match:
                input_str = input_match.group(1).strip()

                # Try to parse as JSON
                try:
                    if input_str.startswith("{"):
                        arguments = json.loads(input_str)
                    elif input_str.startswith('"'):
                        # Single string argument - common case
                        arguments = {"input": json.loads(input_str)}
                    else:
                        # Plain text argument
                        arguments = {"input": input_str}
                except json.JSONDecodeError:
                    # Fallback: treat as single string argument
                    arguments = {"input": input_str}

            tool_call = ToolCall(
                id=str(uuid.uuid4()),
                name=action_name,
                arguments=arguments
            )

        return tool_call, thought, final_answer

    @staticmethod
    def format_observation(tool_name: str, result: str, success: bool = True) -> str:
        """
        Format a tool result as an observation.

        Args:
            tool_name: Name of the tool
            result: Result or error message
            success: Whether the tool succeeded

        Returns:
            Formatted observation string
        """
        if success:
            return f"Observation [{tool_name}]: {result}"
        else:
            return f"Observation [{tool_name}]: Error - {result}"

    @staticmethod
    def is_final_answer(response: str) -> bool:
        """Check if the response contains a final answer"""
        return bool(re.search(r"Final Answer:", response, re.IGNORECASE))

    @staticmethod
    def extract_final_answer(response: str) -> Optional[str]:
        """Extract the final answer from a response"""
        match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
