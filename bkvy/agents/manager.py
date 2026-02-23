"""
Agent manager - manages agent templates and lifecycle
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime

from .executor import AgentExecutor
from .memory import ConversationMemory
from ..models.agent_schemas import (
    AgentType, AgentConfig, AgentSession, AgentRunRequest, AgentRunResponse
)
from ..tools.registry import ToolRegistry
from ..utils.logging import setup_logging

if TYPE_CHECKING:
    from ..core.router import IntelligentRouter
    from ..mcp.manager import MCPManager

logger = setup_logging()

# Global manager instance
_agent_manager: Optional["AgentManager"] = None


class AgentManager:
    """
    Manages agent templates and orchestrates agent execution.
    Singleton pattern matching existing bkvy managers.
    """

    def __init__(
        self,
        router: "IntelligentRouter",
        tool_registry: ToolRegistry,
        mcp_manager: Optional["MCPManager"] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the agent manager.

        Args:
            router: The IntelligentRouter for LLM calls
            tool_registry: Registry of available tools
            mcp_manager: Optional MCP manager
            config_path: Path to agents.json config file
        """
        self.router = router
        self.tool_registry = tool_registry
        self.mcp_manager = mcp_manager
        self.config_path = config_path or "config/agents.json"

        self._templates: Dict[str, AgentConfig] = {}
        self._active_sessions: Dict[str, AgentSession] = {}

        # Create executor
        self.executor = AgentExecutor(router, tool_registry, mcp_manager)

    async def load_templates(self) -> None:
        """Load agent templates from config file"""
        config_file = Path(self.config_path)

        if not config_file.exists():
            logger.warning("Agent config file not found, using defaults", path=self.config_path)
            self._load_default_templates()
            return

        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            templates = config.get("templates", {})
            for name, template_data in templates.items():
                try:
                    # Parse agent type
                    agent_type = AgentType(template_data.get("agent_type", "react"))

                    template = AgentConfig(
                        name=name,
                        agent_type=agent_type,
                        intelligence_level=template_data.get("intelligence_level", "medium"),
                        max_iterations=template_data.get("max_iterations", 10),
                        tools=template_data.get("tools", []),
                        system_prompt=template_data.get("system_prompt"),
                        temperature=template_data.get("temperature"),
                        max_tokens=template_data.get("max_tokens")
                    )

                    self._templates[name] = template
                    logger.info("Loaded agent template", name=name)

                except Exception as e:
                    logger.error("Failed to load template", name=name, error=str(e))

            logger.info("Loaded agent templates", count=len(self._templates))

        except Exception as e:
            logger.error("Failed to load agent config", path=self.config_path, error=str(e))
            self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default agent templates"""
        self._templates = {
            "general": AgentConfig(
                name="general",
                agent_type=AgentType.REACT,
                intelligence_level="medium",
                max_iterations=10,
                tools=["web_fetch", "web_search"]
            ),
            "research": AgentConfig(
                name="research",
                agent_type=AgentType.REACT,
                intelligence_level="high",
                max_iterations=15,
                tools=["web_fetch", "web_search"],
                system_prompt="You are a research assistant. Be thorough and cite sources."
            ),
            "simple": AgentConfig(
                name="simple",
                agent_type=AgentType.FUNCTION_CALLING,
                intelligence_level="low",
                max_iterations=5,
                tools=["web_fetch"]
            )
        }
        logger.info("Loaded default agent templates", count=len(self._templates))

    def register_template(self, template: AgentConfig) -> None:
        """Register a new agent template"""
        self._templates[template.name] = template
        logger.info("Registered agent template", name=template.name)

    def unregister_template(self, name: str) -> bool:
        """Unregister an agent template"""
        if name in self._templates:
            del self._templates[name]
            logger.info("Unregistered agent template", name=name)
            return True
        return False

    def get_template(self, name: str) -> Optional[AgentConfig]:
        """Get an agent template by name"""
        return self._templates.get(name)

    def list_templates(self) -> List[AgentConfig]:
        """List all available templates"""
        return list(self._templates.values())

    def list_template_names(self) -> List[str]:
        """List all template names"""
        return list(self._templates.keys())

    async def run_agent(self, request: AgentRunRequest) -> AgentRunResponse:
        """
        Run an agent task.

        Args:
            request: The run request

        Returns:
            Agent run response
        """
        # Get template
        template = self.get_template(request.agent_template)
        if not template:
            return AgentRunResponse(
                success=False,
                session_id="",
                status="failed",
                error=f"Template not found: {request.agent_template}"
            )

        # Create config (with overrides)
        config = AgentConfig(
            name=template.name,
            agent_type=template.agent_type,
            intelligence_level=template.intelligence_level,
            max_iterations=request.max_iterations or template.max_iterations,
            tools=request.tools or template.tools,
            system_prompt=template.system_prompt,
            temperature=template.temperature,
            max_tokens=template.max_tokens
        )

        # Create session
        session_id = str(uuid.uuid4())
        session = AgentSession(
            session_id=session_id,
            client_id=request.client_id,
            template_name=request.agent_template,
            config=config,
            task=request.task,
            created_at=datetime.utcnow()
        )

        # Store active session
        self._active_sessions[session_id] = session

        logger.info(
            "Starting agent run",
            session_id=session_id,
            client_id=request.client_id,
            template=request.agent_template,
            task_length=len(request.task)
        )

        try:
            # Run the agent
            session = await self.executor.run(session, request.context)

            # Update stored session
            self._active_sessions[session_id] = session

            # Build response
            response = AgentRunResponse(
                success=session.status == "completed",
                session_id=session_id,
                response=session.final_response,
                status=session.status,
                iterations=len(session.steps),
                tool_calls=session.total_tool_calls,
                total_tokens=session.total_tokens_used,
                steps=session.steps if request.debug else None,
                error=session.error
            )

            logger.info(
                "Agent run completed",
                session_id=session_id,
                status=session.status,
                iterations=len(session.steps),
                tool_calls=session.total_tool_calls
            )

            return response

        except Exception as e:
            logger.error(
                "Agent run failed",
                session_id=session_id,
                error=str(e)
            )

            return AgentRunResponse(
                success=False,
                session_id=session_id,
                status="failed",
                error=str(e)
            )

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get an active session by ID"""
        return self._active_sessions.get(session_id)

    def list_active_sessions(self) -> List[str]:
        """List active session IDs"""
        return list(self._active_sessions.keys())

    def cleanup_session(self, session_id: str) -> bool:
        """Remove a completed session"""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            return True
        return False


def get_agent_manager() -> Optional[AgentManager]:
    """Get the global agent manager singleton"""
    return _agent_manager


def init_agent_manager(
    router: "IntelligentRouter",
    tool_registry: ToolRegistry,
    mcp_manager: Optional["MCPManager"] = None,
    config_path: Optional[str] = None
) -> AgentManager:
    """Initialize the global agent manager"""
    global _agent_manager

    _agent_manager = AgentManager(
        router=router,
        tool_registry=tool_registry,
        mcp_manager=mcp_manager,
        config_path=config_path
    )

    logger.info("Agent manager initialized")
    return _agent_manager
