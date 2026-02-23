"""
Pydantic schemas for agent system
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class AgentType(str, Enum):
    """Types of agents supported"""
    REACT = "react"
    FUNCTION_CALLING = "function_calling"


class ToolParameterType(str, Enum):
    """JSON Schema types for tool parameters"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Definition of a tool parameter"""
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., description="Parameter name")
    type: ToolParameterType = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not required")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values for enum types")
    items: Optional[Dict[str, Any]] = Field(default=None, description="Schema for array items")


class ToolDefinition(BaseModel):
    """Definition of a tool for LLM consumption"""
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Tool description for the LLM")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    source: str = Field(default="builtin", description="Tool source: builtin, mcp, custom")
    mcp_server: Optional[str] = Field(default=None, description="MCP server name if source is mcp")

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type.value,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema()
            }
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.to_json_schema()
        }


class ToolCall(BaseModel):
    """A tool call request"""
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(..., description="Unique call identifier")
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ToolResult(BaseModel):
    """Result of a tool execution"""
    model_config = ConfigDict(protected_namespaces=())

    call_id: str = Field(..., description="Corresponding call ID")
    tool_name: str = Field(..., description="Tool name")
    success: bool = Field(..., description="Whether execution succeeded")
    result: Optional[Any] = Field(default=None, description="Result if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")


class AgentConfig(BaseModel):
    """Configuration for an agent template"""
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., description="Template name")
    agent_type: AgentType = Field(..., description="Agent type")
    intelligence_level: str = Field(default="medium", description="Intelligence level for LLM routing")
    max_iterations: int = Field(default=10, description="Maximum think-act-observe iterations")
    tools: List[str] = Field(default_factory=list, description="Tool names to enable")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt override")
    temperature: Optional[float] = Field(default=None, description="Temperature for LLM calls")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for LLM calls")


class AgentStep(BaseModel):
    """A single step in agent execution"""
    model_config = ConfigDict(protected_namespaces=())

    step_number: int = Field(..., description="Step number in sequence")
    thought: Optional[str] = Field(default=None, description="Agent's reasoning (ReAct)")
    action: Optional[ToolCall] = Field(default=None, description="Tool call if any")
    observation: Optional[str] = Field(default=None, description="Tool result observation")
    response: Optional[str] = Field(default=None, description="Final response if complete")
    raw_llm_response: Optional[str] = Field(default=None, description="Raw LLM output")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Step timestamp")


class AgentSession(BaseModel):
    """Active agent session state"""
    model_config = ConfigDict(protected_namespaces=())

    session_id: str = Field(..., description="Unique session identifier")
    client_id: str = Field(..., description="Client identifier")
    template_name: str = Field(..., description="Agent template used")
    config: AgentConfig = Field(..., description="Agent configuration")
    task: str = Field(..., description="Original task/prompt")
    steps: List[AgentStep] = Field(default_factory=list, description="Execution steps")
    status: str = Field(default="running", description="Status: running, completed, failed, max_iterations")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")
    final_response: Optional[str] = Field(default=None, description="Final agent response")
    error: Optional[str] = Field(default=None, description="Error if failed")
    total_llm_calls: int = Field(default=0, description="Number of LLM calls made")
    total_tool_calls: int = Field(default=0, description="Number of tool calls made")
    total_tokens_used: int = Field(default=0, description="Total tokens consumed")


class AgentRunRequest(BaseModel):
    """Request to run an agent task"""
    model_config = ConfigDict(protected_namespaces=())

    client_id: str = Field(..., description="Client identifier for rate limiting")
    task: str = Field(..., description="Task/prompt for the agent")
    agent_template: str = Field(default="general", description="Agent template to use")
    tools: Optional[List[str]] = Field(default=None, description="Override template tools")
    max_iterations: Optional[int] = Field(default=None, description="Override max iterations")
    context: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation context")
    debug: bool = Field(default=False, description="Return detailed debug info")


class AgentRunResponse(BaseModel):
    """Response from agent run"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="Whether agent completed successfully")
    session_id: str = Field(..., description="Session identifier")
    response: Optional[str] = Field(default=None, description="Agent's final response")
    status: str = Field(..., description="Session status")
    iterations: int = Field(default=0, description="Number of iterations executed")
    tool_calls: int = Field(default=0, description="Number of tool calls made")
    total_tokens: int = Field(default=0, description="Total tokens used")
    steps: Optional[List[AgentStep]] = Field(default=None, description="Detailed steps if debug=True")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# MCP (Model Context Protocol) schemas

class MCPTransportType(str, Enum):
    """MCP transport types"""
    STDIO = "stdio"
    HTTP = "http"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server"""
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., description="Server name/identifier")
    transport: MCPTransportType = Field(..., description="Transport type")
    # Stdio transport options
    command: Optional[str] = Field(default=None, description="Command to run for stdio transport")
    args: Optional[List[str]] = Field(default=None, description="Arguments for command")
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")
    # HTTP transport options
    url: Optional[str] = Field(default=None, description="URL for HTTP transport")
    headers: Optional[Dict[str, str]] = Field(default=None, description="HTTP headers")
    # Common options
    auto_start: bool = Field(default=False, description="Auto-start on application startup")
    timeout_seconds: int = Field(default=30, description="Connection timeout")


class MCPServerStatus(BaseModel):
    """Status of an MCP server connection"""
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., description="Server name")
    connected: bool = Field(..., description="Whether connected")
    transport: MCPTransportType = Field(..., description="Transport type")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    connected_at: Optional[datetime] = Field(default=None, description="Connection time")
    error: Optional[str] = Field(default=None, description="Last error if any")


class MCPConnectRequest(BaseModel):
    """Request to connect to an MCP server"""
    model_config = ConfigDict(protected_namespaces=())

    server_name: str = Field(..., description="Server name from config or new config")
    config: Optional[MCPServerConfig] = Field(default=None, description="Override config for new server")


class MCPConnectResponse(BaseModel):
    """Response from MCP connect"""
    model_config = ConfigDict(protected_namespaces=())

    success: bool = Field(..., description="Whether connection succeeded")
    server_name: str = Field(..., description="Server name")
    tools_discovered: List[str] = Field(default_factory=list, description="Tools discovered")
    error: Optional[str] = Field(default=None, description="Error if failed")


class MCPToolCallRequest(BaseModel):
    """Request to call an MCP tool"""
    model_config = ConfigDict(protected_namespaces=())

    server_name: str = Field(..., description="MCP server name")
    tool_name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
