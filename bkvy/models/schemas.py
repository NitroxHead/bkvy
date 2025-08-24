"""
Pydantic schemas for request/response models
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

from .enums import IntelligenceLevel, RoutingMethod


class Message(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    role: str = Field(..., description="Role: user, assistant, system")
    content: str = Field(..., description="Message content")


class LLMOptions(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Temperature for randomness")
    top_p: Optional[float] = Field(None, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(None, description="Top-k sampling parameter")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    disable_thinking: Optional[bool] = Field(None, description="Disable thinking for models that support it")


class IntelligenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    client_id: str = Field(..., description="Client identifier")
    intelligence_level: IntelligenceLevel = Field(..., description="Required intelligence level")
    max_wait_seconds: int = Field(..., description="Maximum estimated wait time for routing (calculation only)")
    messages: List[Message] = Field(..., description="Conversation messages")
    options: Optional[LLMOptions] = Field(None, description="Generation options")
    debug: Optional[bool] = Field(False, description="Return detailed debug information")


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    client_id: str = Field(..., description="Client identifier")
    scenario: str = Field(..., description="Scenario name from routing.json")
    max_wait_seconds: int = Field(..., description="Maximum estimated wait time for routing (calculation only)")
    messages: List[Message] = Field(..., description="Conversation messages")
    options: Optional[LLMOptions] = Field(None, description="Generation options")
    debug: Optional[bool] = Field(False, description="Return detailed debug information")


class DirectRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    client_id: str = Field(..., description="Client identifier")
    provider: str = Field(..., description="Provider name (gemini, openai, anthropic)")
    model_name: str = Field(..., description="Model name", alias="model")
    api_key_id: Optional[str] = Field(None, description="Specific API key ID (auto-select if None)")
    max_wait_seconds: int = Field(..., description="Maximum estimated wait time for routing (calculation only)")
    messages: List[Message] = Field(..., description="Conversation messages")
    options: Optional[LLMOptions] = Field(None, description="Generation options")
    debug: Optional[bool] = Field(False, description="Return detailed debug information")


class UsageInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponseMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    rate_limit_wait_ms: int
    queue_wait_ms: int
    api_response_time_ms: int
    total_completion_time_ms: int
    cost_usd: float
    alternatives_considered: List[Dict[str, Any]]


class SimplifiedResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool
    request_id: str
    model_used: Optional[str] = None
    content: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    truncated: Optional[bool] = None
    error: Optional[str] = None


class LLMResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool
    request_id: str
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    api_key_used: Optional[str] = None
    routing_method: Optional[RoutingMethod] = None
    decision_reason: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    metadata: Optional[ResponseMetadata] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    retry_suggestion: Optional[Dict[str, Any]] = None
    evaluated_combinations: Optional[List[Dict[str, Any]]] = None