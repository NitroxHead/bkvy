"""
Agent executor - orchestrates the think-act-observe loop
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .memory import ConversationMemory
from .prompts.react import ReActPrompts
from .prompts.function_calling import FunctionCallingPrompts
from ..models.agent_schemas import (
    AgentType, AgentConfig, AgentSession, AgentStep,
    ToolCall, ToolResult, ToolDefinition
)
from ..models.schemas import IntelligenceRequest, Message, LLMOptions
from ..models.enums import IntelligenceLevel
from ..tools.registry import ToolRegistry
from ..utils.logging import setup_logging

if TYPE_CHECKING:
    from ..core.router import IntelligentRouter
    from ..mcp.manager import MCPManager

logger = setup_logging()


class AgentExecutor:
    """
    Executes agent tasks using the think-act-observe pattern.
    Routes all LLM calls through the existing IntelligentRouter.
    """

    def __init__(
        self,
        router: "IntelligentRouter",
        tool_registry: ToolRegistry,
        mcp_manager: Optional["MCPManager"] = None
    ):
        """
        Initialize the executor.

        Args:
            router: The IntelligentRouter for LLM calls
            tool_registry: Registry of available tools
            mcp_manager: Optional MCP manager for MCP tools
        """
        self.router = router
        self.tool_registry = tool_registry
        self.mcp_manager = mcp_manager

    async def run(
        self,
        session: AgentSession,
        context: Optional[List[Dict[str, str]]] = None
    ) -> AgentSession:
        """
        Run an agent session to completion.

        Args:
            session: The agent session to run
            context: Optional previous conversation context

        Returns:
            Updated session with results
        """
        config = session.config

        # Initialize memory
        memory = ConversationMemory(max_tokens=100000)

        # Get tool definitions
        tool_definitions = self.tool_registry.list_by_names(config.tools)

        # Set up system prompt based on agent type
        if config.agent_type == AgentType.REACT:
            system_prompt = ReActPrompts.get_system_prompt(
                tools=tool_definitions,
                custom_instructions=config.system_prompt
            )
        else:
            system_prompt = FunctionCallingPrompts.get_system_prompt(
                custom_instructions=config.system_prompt
            )

        memory.system_prompt = system_prompt

        # Add context if provided
        if context:
            for msg in context:
                memory.add_message(msg["role"], msg["content"])

        # Add the task as user message
        memory.add_user_message(session.task)

        logger.info(
            "Starting agent execution",
            session_id=session.session_id,
            agent_type=config.agent_type.value,
            max_iterations=config.max_iterations,
            tools=config.tools
        )

        # Run the appropriate loop
        try:
            if config.agent_type == AgentType.REACT:
                session = await self._run_react_loop(session, memory, tool_definitions)
            else:
                session = await self._run_function_calling_loop(session, memory, tool_definitions)

        except Exception as e:
            logger.error(
                "Agent execution failed",
                session_id=session.session_id,
                error=str(e)
            )
            session.status = "failed"
            session.error = str(e)

        session.completed_at = datetime.utcnow()
        return session

    async def _run_react_loop(
        self,
        session: AgentSession,
        memory: ConversationMemory,
        tools: List[ToolDefinition]
    ) -> AgentSession:
        """Run the ReAct think-act-observe loop"""

        config = session.config

        for iteration in range(config.max_iterations):
            step_number = iteration + 1

            logger.info(
                "ReAct iteration",
                session_id=session.session_id,
                iteration=step_number
            )

            # Call LLM
            llm_response = await self._call_llm(
                session=session,
                memory=memory,
                tools=None  # ReAct uses text-based tools
            )

            if not llm_response:
                session.status = "failed"
                session.error = "LLM call failed"
                return session

            response_text = llm_response.get("content", "")

            # Parse the response
            tool_call, thought, final_answer = ReActPrompts.parse_action(response_text)

            # Create step record
            step = AgentStep(
                step_number=step_number,
                thought=thought,
                raw_llm_response=response_text,
                timestamp=datetime.utcnow()
            )

            # Check for final answer
            if final_answer:
                step.response = final_answer
                session.steps.append(step)
                session.final_response = final_answer
                session.status = "completed"
                memory.add_assistant_message(response_text)
                logger.info(
                    "ReAct completed with final answer",
                    session_id=session.session_id,
                    iterations=step_number
                )
                return session

            # Handle tool call
            if tool_call:
                step.action = tool_call

                # Execute the tool
                result = await self.tool_registry.execute(tool_call)
                session.total_tool_calls += 1

                # Format observation
                if result.success:
                    observation = str(result.result)
                else:
                    observation = f"Error: {result.error}"

                step.observation = observation

                # Add to memory
                memory.add_assistant_message(response_text)
                memory.add_message(
                    "user",
                    ReActPrompts.format_observation(tool_call.name, observation, result.success)
                )

                logger.info(
                    "Tool executed",
                    session_id=session.session_id,
                    tool=tool_call.name,
                    success=result.success
                )
            else:
                # No tool call and no final answer - might be intermediate reasoning
                memory.add_assistant_message(response_text)

            session.steps.append(step)

        # Max iterations reached
        session.status = "max_iterations"
        session.error = f"Reached maximum iterations ({config.max_iterations})"

        # Try to extract any partial answer
        last_response = memory.get_last_assistant_message()
        if last_response:
            session.final_response = last_response.content

        logger.warning(
            "ReAct reached max iterations",
            session_id=session.session_id,
            max_iterations=config.max_iterations
        )

        return session

    async def _run_function_calling_loop(
        self,
        session: AgentSession,
        memory: ConversationMemory,
        tools: List[ToolDefinition]
    ) -> AgentSession:
        """Run the native function calling loop"""

        config = session.config

        for iteration in range(config.max_iterations):
            step_number = iteration + 1

            logger.info(
                "Function calling iteration",
                session_id=session.session_id,
                iteration=step_number
            )

            # Call LLM with tools
            llm_response = await self._call_llm(
                session=session,
                memory=memory,
                tools=tools
            )

            if not llm_response:
                session.status = "failed"
                session.error = "LLM call failed"
                return session

            raw_response = llm_response.get("raw_response", {})
            content = llm_response.get("content", "")

            # Determine provider format from response structure
            is_anthropic = "content" in raw_response and isinstance(raw_response.get("content"), list)

            # Create step record
            step = AgentStep(
                step_number=step_number,
                raw_llm_response=content,
                timestamp=datetime.utcnow()
            )

            # Check for tool calls
            if is_anthropic:
                has_tools = FunctionCallingPrompts.has_tool_calls_anthropic(raw_response)
                tool_calls = FunctionCallingPrompts.parse_tool_calls_anthropic(raw_response) if has_tools else []
                text_content = FunctionCallingPrompts.get_text_content_anthropic(raw_response)
                is_complete = FunctionCallingPrompts.is_stop_reason_anthropic(raw_response)
            else:
                has_tools = FunctionCallingPrompts.has_tool_calls_openai(raw_response)
                tool_calls = FunctionCallingPrompts.parse_tool_calls_openai(raw_response) if has_tools else []
                text_content = FunctionCallingPrompts.get_text_content_openai(raw_response)
                is_complete = FunctionCallingPrompts.is_stop_reason_openai(raw_response)

            # If no tool calls and complete, we're done
            if not tool_calls and is_complete:
                step.response = text_content or content
                session.steps.append(step)
                session.final_response = step.response
                session.status = "completed"
                memory.add_assistant_message(step.response)
                logger.info(
                    "Function calling completed",
                    session_id=session.session_id,
                    iterations=step_number
                )
                return session

            # Process tool calls
            if tool_calls:
                step.thought = text_content

                # Execute all tool calls
                tool_results = []
                for tc in tool_calls:
                    step.action = tc  # Last one recorded in step

                    result = await self.tool_registry.execute(tc)
                    session.total_tool_calls += 1
                    tool_results.append((tc, result))

                    logger.info(
                        "Tool executed",
                        session_id=session.session_id,
                        tool=tc.name,
                        success=result.success
                    )

                # Format observations
                observations = []
                for tc, result in tool_results:
                    if result.success:
                        observations.append(f"{tc.name}: {result.result}")
                    else:
                        observations.append(f"{tc.name}: Error - {result.error}")

                step.observation = "\n".join(observations)

                # Add to memory - format depends on provider
                # For simplicity, we add as user message with tool results
                memory.add_assistant_message(content or text_content or "")

                for tc, result in tool_results:
                    result_str = str(result.result) if result.success else f"Error: {result.error}"
                    memory.add_tool_result(tc.id, result_str, {"tool_name": tc.name})

            session.steps.append(step)

        # Max iterations reached
        session.status = "max_iterations"
        session.error = f"Reached maximum iterations ({config.max_iterations})"

        last_response = memory.get_last_assistant_message()
        if last_response:
            session.final_response = last_response.content

        logger.warning(
            "Function calling reached max iterations",
            session_id=session.session_id,
            max_iterations=config.max_iterations
        )

        return session

    async def _call_llm(
        self,
        session: AgentSession,
        memory: ConversationMemory,
        tools: Optional[List[ToolDefinition]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make an LLM call through the IntelligentRouter.

        Args:
            session: Current agent session
            memory: Conversation memory
            tools: Optional tool definitions for function calling

        Returns:
            LLM response dict or None on failure
        """
        config = session.config

        # Format messages
        messages = memory.format_for_llm("openai")

        # Convert to Message objects
        message_objs = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]

        # Build options
        options = LLMOptions()
        if config.temperature is not None:
            options.temperature = config.temperature
        if config.max_tokens is not None:
            options.max_tokens = config.max_tokens

        # Add tools to options if provided
        options_dict = options.model_dump(exclude_none=True)
        if tools:
            options_dict["tools"] = self.tool_registry.to_openai_tools(config.tools)

        # Map intelligence level string to enum
        try:
            intel_level = IntelligenceLevel(config.intelligence_level)
        except ValueError:
            intel_level = IntelligenceLevel.MEDIUM

        # Build request
        request = IntelligenceRequest(
            client_id=session.client_id,
            intelligence_level=intel_level,
            max_wait_seconds=60,
            messages=message_objs,
            options=LLMOptions(**options_dict) if not tools else None,
            debug=True
        )

        try:
            response = await self.router.route_intelligence_request(request)

            session.total_llm_calls += 1

            if response.success:
                # Extract usage
                usage = response.response.get("usage", {}) if response.response else {}
                session.total_tokens_used += usage.get("total_tokens", 0)

                return response.response
            else:
                logger.error(
                    "LLM call failed",
                    session_id=session.session_id,
                    error=response.message
                )
                return None

        except Exception as e:
            logger.error(
                "LLM call exception",
                session_id=session.session_id,
                error=str(e)
            )
            return None
