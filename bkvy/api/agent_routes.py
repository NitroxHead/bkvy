"""
API routes for agent system
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException

from ..models.agent_schemas import (
    AgentConfig, AgentRunRequest, AgentRunResponse,
    ToolDefinition, MCPServerConfig, MCPConnectRequest, MCPConnectResponse,
    MCPServerStatus
)
from ..agents.manager import get_agent_manager
from ..tools.registry import get_tool_registry
from ..mcp.manager import get_mcp_manager

# =============================================================================
# AGENT ROUTES
# =============================================================================

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest):
    """
    Run an agent task.

    The agent will use tools and LLM calls to complete the task,
    returning the final response when done.
    """
    agent_manager = get_agent_manager()

    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    response = await agent_manager.run_agent(request)
    return response


@router.get("/templates", response_model=List[AgentConfig])
async def list_templates():
    """List all available agent templates"""
    agent_manager = get_agent_manager()

    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    return agent_manager.list_templates()


@router.get("/templates/{name}", response_model=AgentConfig)
async def get_template(name: str):
    """Get a specific agent template"""
    agent_manager = get_agent_manager()

    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    template = agent_manager.get_template(name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")

    return template


@router.post("/templates", response_model=AgentConfig)
async def create_template(template: AgentConfig):
    """Create or update an agent template"""
    agent_manager = get_agent_manager()

    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    agent_manager.register_template(template)
    return template


@router.delete("/templates/{name}")
async def delete_template(name: str):
    """Delete an agent template"""
    agent_manager = get_agent_manager()

    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent system not initialized")

    success = agent_manager.unregister_template(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Template not found: {name}")

    return {"success": True, "message": f"Template '{name}' deleted"}


# =============================================================================
# TOOL ROUTES
# =============================================================================

tools_router = APIRouter(prefix="/tools", tags=["Tools"])


@tools_router.get("", response_model=List[ToolDefinition])
async def list_tools():
    """List all available tools"""
    tool_registry = get_tool_registry()

    if not tool_registry:
        raise HTTPException(status_code=503, detail="Tool system not initialized")

    return tool_registry.list_all()


@tools_router.get("/builtin", response_model=List[str])
async def list_builtin_tools():
    """List built-in tool names"""
    tool_registry = get_tool_registry()

    if not tool_registry:
        raise HTTPException(status_code=503, detail="Tool system not initialized")

    return tool_registry.list_builtin_names()


@tools_router.get("/mcp", response_model=List[str])
async def list_mcp_tools():
    """List MCP tool names"""
    tool_registry = get_tool_registry()

    if not tool_registry:
        raise HTTPException(status_code=503, detail="Tool system not initialized")

    return tool_registry.list_mcp_names()


@tools_router.get("/{name}", response_model=ToolDefinition)
async def get_tool(name: str):
    """Get details for a specific tool"""
    tool_registry = get_tool_registry()

    if not tool_registry:
        raise HTTPException(status_code=503, detail="Tool system not initialized")

    definition = tool_registry.get_definition(name)
    if not definition:
        raise HTTPException(status_code=404, detail=f"Tool not found: {name}")

    return definition


# =============================================================================
# MCP ROUTES
# =============================================================================

mcp_router = APIRouter(prefix="/mcp", tags=["MCP"])


@mcp_router.get("/status", response_model=Dict[str, MCPServerStatus])
async def get_mcp_status():
    """Get status of all MCP servers"""
    mcp_manager = get_mcp_manager()

    if not mcp_manager:
        return {}

    return mcp_manager.get_all_statuses()


@mcp_router.get("/status/{name}", response_model=MCPServerStatus)
async def get_mcp_server_status(name: str):
    """Get status of a specific MCP server"""
    mcp_manager = get_mcp_manager()

    if not mcp_manager:
        raise HTTPException(status_code=503, detail="MCP system not initialized")

    return mcp_manager.get_status(name)


@mcp_router.post("/connect", response_model=MCPConnectResponse)
async def connect_mcp_server(request: MCPConnectRequest):
    """Connect to an MCP server"""
    mcp_manager = get_mcp_manager()

    if not mcp_manager:
        raise HTTPException(status_code=503, detail="MCP system not initialized")

    # If config provided, add it first
    if request.config:
        mcp_manager.add_config(request.config)

    success = await mcp_manager.connect(request.server_name, request.config)

    if success:
        status = mcp_manager.get_status(request.server_name)
        return MCPConnectResponse(
            success=True,
            server_name=request.server_name,
            tools_discovered=status.tools
        )
    else:
        status = mcp_manager.get_status(request.server_name)
        return MCPConnectResponse(
            success=False,
            server_name=request.server_name,
            error=status.error or "Connection failed"
        )


@mcp_router.post("/disconnect/{name}")
async def disconnect_mcp_server(name: str):
    """Disconnect from an MCP server"""
    mcp_manager = get_mcp_manager()

    if not mcp_manager:
        raise HTTPException(status_code=503, detail="MCP system not initialized")

    success = await mcp_manager.disconnect(name)

    return {
        "success": success,
        "message": f"Disconnected from {name}" if success else f"Server not connected: {name}"
    }


@mcp_router.get("/configs", response_model=List[str])
async def list_mcp_configs():
    """List all configured MCP servers"""
    mcp_manager = get_mcp_manager()

    if not mcp_manager:
        return []

    return mcp_manager.list_configs()


@mcp_router.get("/configs/{name}", response_model=MCPServerConfig)
async def get_mcp_config(name: str):
    """Get configuration for a specific MCP server"""
    mcp_manager = get_mcp_manager()

    if not mcp_manager:
        raise HTTPException(status_code=503, detail="MCP system not initialized")

    config = mcp_manager.get_config(name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Server config not found: {name}")

    return config


@mcp_router.post("/configs", response_model=MCPServerConfig)
async def add_mcp_config(config: MCPServerConfig):
    """Add or update an MCP server configuration"""
    mcp_manager = get_mcp_manager()

    if not mcp_manager:
        raise HTTPException(status_code=503, detail="MCP system not initialized")

    mcp_manager.add_config(config)
    return config


@mcp_router.delete("/configs/{name}")
async def delete_mcp_config(name: str):
    """Delete an MCP server configuration"""
    mcp_manager = get_mcp_manager()

    if not mcp_manager:
        raise HTTPException(status_code=503, detail="MCP system not initialized")

    # Disconnect first if connected
    if mcp_manager.is_connected(name):
        await mcp_manager.disconnect(name)

    success = mcp_manager.remove_config(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Server config not found: {name}")

    return {"success": True, "message": f"Config '{name}' deleted"}
