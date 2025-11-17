#!/usr/bin/env python3
"""
Advanced Multi-Tier LLM API Router with Intelligent Cost-Time-Rate Optimization

A sophisticated HTTP server that manages complex rate limiting across multiple API keys 
and models, with intelligent routing that dynamically selects optimal providers based 
on real-time queue states, rate limit statuses, costs, and time constraints.

Key Features:
- Individual rate limits per (API_KEY, MODEL) combination
- Intelligent total completion time calculation (rate limit wait + queue wait + processing time)
- Cost-time optimization with automatic failover
- Multiple routing methods: intelligence-based, scenario-based, direct
- Real-time state tracking and monitoring
- Support for Gemini, OpenAI, and Anthropic APIs
- SYNCHRONOUS responses - wait for actual LLM completion
"""

import asyncio
import aiohttp
import aiofiles
import json
import time
import uuid
import structlog
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import uvicorn

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# =============================================================================
# MODELS AND ENUMS
# =============================================================================

class IntelligenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class RoutingMethod(str, Enum):
    INTELLIGENCE = "intelligence"
    SCENARIO = "scenario"
    DIRECT = "direct"

@dataclass
class RateLimitState:
    """Track rate limit state for a specific (API_KEY, MODEL) combination"""
    requests_this_minute: int = 0
    requests_today: int = 0
    minute_reset_time: datetime = None
    day_reset_time: datetime = None
    currently_rate_limited: bool = False
    rate_limit_wait_seconds: float = 0
    rpm_limit: int = 0
    rpd_limit: int = 0
    last_request_time: datetime = None

    def __post_init__(self):
        if self.minute_reset_time is None:
            now = datetime.now(timezone.utc)
            self.minute_reset_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        if self.day_reset_time is None:
            now = datetime.now(timezone.utc)
            self.day_reset_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

@dataclass 
class QueueState:
    """Track queue state for a specific (API_KEY, MODEL) combination"""
    current_queue_length: int = 0
    active_requests: List[str] = None
    estimated_queue_wait_seconds: float = 0
    last_updated: datetime = None

    def __post_init__(self):
        if self.active_requests is None:
            self.active_requests = []
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)

@dataclass
class CompletionTimeAnalysis:
    """Analysis of total completion time for a combination"""
    rate_limit_wait_seconds: float
    queue_wait_seconds: float
    processing_time_seconds: float
    total_seconds: float
    cost_per_1k_tokens: float
    combination_key: str
    provider: str
    model: str
    api_key_id: str

@dataclass
class ProviderModel:
    """Model configuration for a provider"""
    endpoint: str
    cost_per_1k_tokens: float
    avg_response_time_ms: int
    intelligence_tier: str
    version: Optional[str] = None

@dataclass
class ProviderKey:
    """API key configuration with rate limits"""
    api_key: str
    rate_limits: Dict[str, Dict[str, int]]

@dataclass
class ProviderConfig:
    """Complete provider configuration"""
    keys: Dict[str, ProviderKey]
    models: Dict[str, ProviderModel]

# =============================================================================
# PYDANTIC REQUEST/RESPONSE MODELS
# =============================================================================

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

class IntelligenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    client_id: str = Field(..., description="Client identifier")
    intelligence_level: IntelligenceLevel = Field(..., description="Required intelligence level")
    max_wait_seconds: int = Field(..., description="Maximum estimated wait time for routing (calculation only)")
    messages: List[Message] = Field(..., description="Conversation messages")
    options: Optional[LLMOptions] = Field(None, description="Generation options")

class ScenarioRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    client_id: str = Field(..., description="Client identifier")
    scenario: str = Field(..., description="Scenario name from routing.json")
    max_wait_seconds: int = Field(..., description="Maximum estimated wait time for routing (calculation only)")
    messages: List[Message] = Field(..., description="Conversation messages")
    options: Optional[LLMOptions] = Field(None, description="Generation options")

class DirectRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    client_id: str = Field(..., description="Client identifier")
    provider: str = Field(..., description="Provider name (gemini, openai, anthropic)")
    model_name: str = Field(..., description="Model name", alias="model")
    api_key_id: Optional[str] = Field(None, description="Specific API key ID (auto-select if None)")
    max_wait_seconds: int = Field(..., description="Maximum estimated wait time for routing (calculation only)")
    messages: List[Message] = Field(..., description="Conversation messages")
    options: Optional[LLMOptions] = Field(None, description="Generation options")

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

# =============================================================================
# CONFIGURATION MANAGER
# =============================================================================

class ConfigManager:
    """Manages loading and hot-reloading of configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.providers: Dict[str, ProviderConfig] = {}
        self.scenarios: Dict[str, List[Dict[str, Any]]] = {}
        self._last_providers_mtime = 0
        self._last_routing_mtime = 0
    
    async def load_configs(self):
        """Load all configuration files"""
        await self._load_providers()
        await self._load_scenarios()
    
    async def _load_providers(self):
        """Load providers.json configuration"""
        providers_file = self.config_dir / "providers.json"
        
        if not providers_file.exists():
            logger.error("providers.json not found", file_path=str(providers_file))
            raise FileNotFoundError(f"Configuration file not found: {providers_file}")
        
        current_mtime = providers_file.stat().st_mtime
        if current_mtime == self._last_providers_mtime:
            return  # No changes
        
        try:
            async with aiofiles.open(providers_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            # Parse provider configurations
            for provider_name, provider_data in data.items():
                keys = {}
                for key_id, key_data in provider_data.get("keys", {}).items():
                    keys[key_id] = ProviderKey(
                        api_key=key_data["api_key"],
                        rate_limits=key_data["rate_limits"]
                    )
                
                models = {}
                for model_name, model_data in provider_data.get("models", {}).items():
                    models[model_name] = ProviderModel(
                        endpoint=model_data["endpoint"],
                        cost_per_1k_tokens=model_data["cost_per_1k_tokens"],
                        avg_response_time_ms=model_data["avg_response_time_ms"],
                        intelligence_tier=model_data["intelligence_tier"],
                        version=model_data.get("version")
                    )
                
                self.providers[provider_name] = ProviderConfig(keys=keys, models=models)
            
            self._last_providers_mtime = current_mtime
            logger.info("Loaded providers configuration", provider_count=len(self.providers))
            
        except Exception as e:
            logger.error("Failed to load providers configuration", error=str(e))
            raise
    
    async def _load_scenarios(self):
        """Load routing.json configuration"""
        routing_file = self.config_dir / "routing.json"
        
        if not routing_file.exists():
            logger.error("routing.json not found", file_path=str(routing_file))
            raise FileNotFoundError(f"Configuration file not found: {routing_file}")
        
        current_mtime = routing_file.stat().st_mtime
        if current_mtime == self._last_routing_mtime:
            return  # No changes
        
        try:
            async with aiofiles.open(routing_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            self.scenarios = data.get("scenarios", {})
            self._last_routing_mtime = current_mtime
            logger.info("Loaded routing scenarios", scenario_count=len(self.scenarios))
            
        except Exception as e:
            logger.error("Failed to load routing configuration", error=str(e))
            raise
    
    async def refresh_if_changed(self):
        """Check for configuration file changes and reload if necessary"""
        await self._load_providers()
        await self._load_scenarios()
    
    def get_models_by_intelligence(self, intelligence_level: str) -> List[Tuple[str, str]]:
        """Get all (provider, model) combinations for an intelligence level"""
        combinations = []
        
        logger.info("🔍 CONFIG DEBUG: Looking for models with intelligence level", 
                   intelligence_level=intelligence_level)
        
        for provider_name, config in self.providers.items():
            logger.info("🔍 CONFIG DEBUG: Checking provider", 
                       provider=provider_name,
                       model_count=len(config.models))
            
            for model_name, model in config.models.items():
                logger.info("🔍 CONFIG DEBUG: Checking model", 
                           provider=provider_name,
                           model=model_name,
                           model_intelligence_tier=model.intelligence_tier,
                           matches=model.intelligence_tier == intelligence_level)
                
                if model.intelligence_tier == intelligence_level:
                    combinations.append((provider_name, model_name))
                    logger.info("✅ CONFIG DEBUG: Model added to combinations", 
                               provider=provider_name,
                               model=model_name,
                               intelligence_tier=model.intelligence_tier)
        
        logger.info("🏁 CONFIG DEBUG: Intelligence level search complete", 
                   intelligence_level=intelligence_level,
                   total_combinations=len(combinations),
                   combinations=combinations)
        
        return combinations
    
    def get_scenario_combinations(self, scenario_name: str) -> List[Tuple[str, str, int]]:
        """Get (provider, model, priority) combinations for a scenario"""
        if scenario_name not in self.scenarios:
            return []
        
        combinations = []
        for item in self.scenarios[scenario_name]:
            combinations.append((item["provider"], item["model"], item["priority"]))
        return combinations

# =============================================================================
# RATE LIMIT MANAGER
# =============================================================================

class RateLimitManager:
    """Manages rate limiting for all (API_KEY, MODEL) combinations"""
    
    def __init__(self, state_dir: str = "rate_states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.states: Dict[str, RateLimitState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
    
    def _get_combination_key(self, provider: str, model: str, api_key_id: str) -> str:
        """Generate unique key for (provider, model, api_key_id) combination"""
        return f"{provider}_{api_key_id}_{model}"
    
    def _get_lock(self, combination_key: str) -> asyncio.Lock:
        """Get or create lock for a combination"""
        if combination_key not in self._locks:
            self._locks[combination_key] = asyncio.Lock()
        return self._locks[combination_key]
    
    async def _load_state(self, combination_key: str) -> RateLimitState:
        """Load rate limit state from file"""
        state_file = self.state_dir / f"{combination_key}.json"
        
        if not state_file.exists():
            return RateLimitState()
        
        try:
            async with aiofiles.open(state_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            # Parse datetime fields
            state = RateLimitState()
            for field, value in data.items():
                if field.endswith('_time') and value:
                    setattr(state, field, datetime.fromisoformat(value))
                else:
                    setattr(state, field, value)
            
            return state
            
        except Exception as e:
            logger.warning("Failed to load rate limit state", 
                         combination=combination_key, error=str(e))
            return RateLimitState()
    
    async def _save_state(self, combination_key: str, state: RateLimitState):
        """Save rate limit state to file"""
        state_file = self.state_dir / f"{combination_key}.json"
        
        try:
            # Convert datetime fields to ISO format
            data = {}
            for field, value in asdict(state).items():
                if field.endswith('_time') and value:
                    data[field] = value.isoformat()
                else:
                    data[field] = value
            
            async with aiofiles.open(state_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logger.error("Failed to save rate limit state", 
                        combination=combination_key, error=str(e))
    
    async def _update_rate_counters(self, state: RateLimitState):
        """Update rate limit counters and reset times"""
        now = datetime.now(timezone.utc)
        
        # Reset minute counter if needed
        if now >= state.minute_reset_time:
            state.requests_this_minute = 0
            state.minute_reset_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Reset daily counter if needed
        if now >= state.day_reset_time:
            state.requests_today = 0
            state.day_reset_time = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    async def check_rate_limit_status(self, provider: str, model: str, api_key_id: str, 
                                    rpm_limit: int, rpd_limit: int) -> Tuple[bool, float]:
        """Check if combination is rate limited and return wait time"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)
        
        async with lock:
            if combination_key not in self.states:
                self.states[combination_key] = await self._load_state(combination_key)
            
            state = self.states[combination_key]
            state.rpm_limit = rpm_limit
            state.rpd_limit = rpd_limit
            
            await self._update_rate_counters(state)
            
            now = datetime.now(timezone.utc)
            
            # Check if rate limited
            if state.requests_this_minute >= rpm_limit:
                wait_seconds = (state.minute_reset_time - now).total_seconds()
                state.currently_rate_limited = True
                state.rate_limit_wait_seconds = max(0, wait_seconds)
                return True, wait_seconds
            
            if state.requests_today >= rpd_limit:
                wait_seconds = (state.day_reset_time - now).total_seconds()
                state.currently_rate_limited = True
                state.rate_limit_wait_seconds = max(0, wait_seconds)
                return True, wait_seconds
            
            state.currently_rate_limited = False
            state.rate_limit_wait_seconds = 0
            return False, 0
    
    async def record_request(self, provider: str, model: str, api_key_id: str):
        """Record a request for rate limiting purposes"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)
        
        async with lock:
            if combination_key not in self.states:
                self.states[combination_key] = await self._load_state(combination_key)
            
            state = self.states[combination_key]
            await self._update_rate_counters(state)
            
            state.requests_this_minute += 1
            state.requests_today += 1
            state.last_request_time = datetime.now(timezone.utc)
            
            await self._save_state(combination_key, state)
    
    async def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all rate limit states for monitoring"""
        states = {}
        for combination_key, state in self.states.items():
            await self._update_rate_counters(state)
            states[combination_key] = asdict(state)
        return states

# =============================================================================
# QUEUE MANAGER - CORRECTED
# =============================================================================

class QueueManager:
    """Manages request queues for all (API_KEY, MODEL) combinations"""
    
    def __init__(self, state_file: str = "queue_states.json"):
        self.state_file = Path(state_file)
        self.queues: Dict[str, asyncio.Queue] = {}
        self.states: Dict[str, QueueState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
    
    def _get_combination_key(self, provider: str, model: str, api_key_id: str) -> str:
        """Generate unique key for (provider, model, api_key_id) combination"""
        return f"{provider}_{api_key_id}_{model}"
    
    def _get_lock(self, combination_key: str) -> asyncio.Lock:
        """Get or create lock for a combination"""
        if combination_key not in self._locks:
            self._locks[combination_key] = asyncio.Lock()
        return self._locks[combination_key]
    
    def _get_queue(self, combination_key: str) -> asyncio.Queue:
        """Get or create queue for a combination"""
        if combination_key not in self.queues:
            self.queues[combination_key] = asyncio.Queue()
            self.states[combination_key] = QueueState()
        return self.queues[combination_key]
    
    async def get_queue_wait_time(self, provider: str, model: str, api_key_id: str, 
                                avg_response_time_ms: int) -> float:
        """Calculate estimated wait time in queue"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        queue = self._get_queue(combination_key)
        state = self.states[combination_key]
        
        queue_length = queue.qsize()
        avg_response_time_seconds = max(avg_response_time_ms / 1000, 0.1)  # Prevent division by zero
        
        estimated_wait = queue_length * avg_response_time_seconds
        
        # Update state
        state.current_queue_length = queue_length
        state.estimated_queue_wait_seconds = estimated_wait
        state.last_updated = datetime.now(timezone.utc)
        
        return estimated_wait
    
    async def execute_request_directly(self, provider: str, model: str, api_key_id: str,
                                     request_data: Dict[str, Any], max_wait_seconds: int, 
                                     rate_limit_manager, config_manager, llm_client) -> Dict[str, Any]:
        """Execute request directly and wait for response (no timeout - completes regardless of time)"""
        combination_key = self._get_combination_key(provider, model, api_key_id)
        lock = self._get_lock(combination_key)
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info("Executing direct request", 
                   combination=combination_key, 
                   request_id=request_id)
        
        try:
            async with lock:
                # Wait for rate limit if needed (but don't timeout the request)
                await self._wait_for_rate_limit(provider, model, api_key_id, 
                                              rate_limit_manager, config_manager)
                
                # Record the request for rate limiting
                await rate_limit_manager.record_request(provider, model, api_key_id)
                
                # Make the actual API call (no timeout - complete regardless of time)
                response = await llm_client._make_api_call(
                    provider=provider,
                    model=model,
                    api_key=request_data["api_key"],
                    messages=request_data["messages"],
                    options=request_data.get("options", {}),
                    endpoint=request_data["endpoint"],
                    version=request_data.get("version")
                )
                
                processing_time = time.time() - start_time
                
                # Return successful result
                result = {
                    "success": True,
                    "request_id": request_id,
                    "provider_used": provider,
                    "model_used": model,
                    "api_key_used": api_key_id,
                    "processing_time_seconds": processing_time,
                    "response": response,
                    "completed_at": datetime.now(timezone.utc).isoformat()
                }
                
                logger.info("Request processed successfully", 
                           request_id=request_id,
                           provider=provider,
                           model=model,
                           processing_time=processing_time)
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error("Request processing failed", 
                        request_id=request_id,
                        error=str(e),
                        traceback=traceback.format_exc())
            
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "processing_time_seconds": processing_time,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
    
    async def _wait_for_rate_limit(self, provider: str, model: str, api_key_id: str, 
                                 rate_limit_manager, config_manager):
        """Wait for rate limit to clear if needed (no timeout)"""
        
        # Get provider config to check rate limits
        provider_config = config_manager.providers[provider]
        key_config = provider_config.keys[api_key_id]
        rate_limits = key_config.rate_limits[model]
        
        while True:
            is_limited, wait_time = await rate_limit_manager.check_rate_limit_status(
                provider, model, api_key_id, rate_limits["rpm"], rate_limits["rpd"]
            )
            
            if not is_limited:
                break
            
            logger.info("Waiting for rate limit", 
                       provider=provider, model=model, api_key_id=api_key_id,
                       wait_seconds=wait_time)
            
            await asyncio.sleep(min(wait_time, 1))  # Sleep in chunks of 1 second max
    
    async def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all queue states for monitoring"""
        states = {}
        for combination_key, state in self.states.items():
            state_dict = asdict(state)
            # Convert datetime to string for JSON serialization
            if state_dict.get("last_updated"):
                state_dict["last_updated"] = state.last_updated.isoformat()
            states[combination_key] = state_dict
        return states

# =============================================================================
# LLM CLIENT - CORRECTED GEMINI PARSING
# =============================================================================

class LLMClient:
    """Handles actual API calls to LLM providers"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def start(self):
        """Start the HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout for individual requests
        )
    
    async def stop(self):
        """Stop the HTTP session"""
        if self.session:
            await self.session.close()
    
    async def _make_api_call(self, provider: str, model: str, api_key: str, 
                           messages: List[Dict], options: Dict, 
                           endpoint: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Make API call to specific provider"""
        
        if provider == "gemini":
            return await self._call_gemini(endpoint, api_key, messages, options)
        elif provider == "openai":
            return await self._call_openai(endpoint, api_key, model, messages, options)
        elif provider == "anthropic":
            return await self._call_anthropic(endpoint, api_key, model, messages, options, version)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _call_gemini(self, endpoint: str, api_key: str, 
                          messages: List[Dict], options: Dict) -> Dict[str, Any]:
        """Call Gemini API with robust response parsing and error handling"""
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append({
                    "parts": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": msg["content"]}]
                })
            elif msg["role"] == "system":
                # Add system message as first user message
                contents.insert(0, {
                    "parts": [{"text": f"System: {msg['content']}"}]
                })
        
        payload = {
            "contents": contents
        }
        
        # Add generation config if options provided
        if options:
            generation_config = {}
            if "max_tokens" in options and options["max_tokens"] is not None:
                # Ensure minimum token count for Gemini
                generation_config["maxOutputTokens"] = max(options["max_tokens"], 50)
            if "temperature" in options and options["temperature"] is not None:
                generation_config["temperature"] = options["temperature"]
            if "top_p" in options and options["top_p"] is not None:
                generation_config["topP"] = options["top_p"]
            if "top_k" in options and options["top_k"] is not None:
                generation_config["topK"] = options["top_k"]
            
            if generation_config:
                payload["generationConfig"] = generation_config
        
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        logger.info("Making Gemini API call", endpoint=endpoint, payload_size=len(str(payload)))
        
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status == 429:
                error_text = await response.text()
                # Handle rate limiting with specific error for retry logic
                raise Exception(f"Gemini API rate limited {response.status}: {error_text}")
            elif response.status != 200:
                error_text = await response.text()
                raise Exception(f"Gemini API error {response.status}: {error_text}")
            
            result = await response.json()
            logger.info("Gemini API response received", has_candidates=bool(result.get("candidates")))
            
            # Extract content with enhanced error handling
            content = ""
            finish_reason = None
            
            if "candidates" in result and result["candidates"]:
                candidate = result["candidates"][0]
                finish_reason = candidate.get("finishReason")
                
                logger.info("Processing candidate", 
                           candidate_keys=list(candidate.keys()),
                           finish_reason=finish_reason)
                
                # Check for content truncation issues
                if finish_reason == "MAX_TOKENS":
                    logger.warning("Gemini response truncated due to MAX_TOKENS", 
                                 candidate=candidate)
                
                # Enhanced content extraction
                if "content" in candidate and isinstance(candidate["content"], dict):
                    content_obj = candidate["content"]
                    
                    # Primary method: content.parts[0].text
                    if "parts" in content_obj and isinstance(content_obj["parts"], list) and content_obj["parts"]:
                        for part in content_obj["parts"]:
                            if isinstance(part, dict) and "text" in part:
                                content = part["text"]
                                logger.info("Extracted content via content.parts.text", 
                                          content_length=len(content))
                                break
                    
                    # Fallback: check if content has direct text
                    if not content and "text" in content_obj:
                        content = content_obj["text"]
                        logger.info("Extracted content via content.text", content_length=len(content))
                
                # Additional fallbacks
                if not content:
                    if "text" in candidate:
                        content = candidate["text"]
                        logger.info("Extracted content via candidate.text", content_length=len(content))
                    elif "message" in candidate:
                        content = str(candidate["message"])
                        logger.info("Extracted content via candidate.message", content_length=len(content))
            
            # Handle empty or problematic responses
            if not content or content.strip() == "":
                error_msg = f"Gemini returned empty content (finish_reason: {finish_reason})"
                logger.error(error_msg, 
                           result_structure={
                               "candidates_count": len(result.get("candidates", [])),
                               "candidate_keys": list(result.get("candidates", [{}])[0].keys()) if result.get("candidates") else [],
                               "finish_reason": finish_reason
                           })
                raise Exception(error_msg)
            
            # Get usage metadata based on example structure
            usage_data = result.get("usageMetadata", {})
            input_tokens = usage_data.get("promptTokenCount", 0)
            output_tokens = usage_data.get("candidatesTokenCount", 0)
            total_tokens = usage_data.get("totalTokenCount", input_tokens + output_tokens)
            
            # Fallback token estimation if not provided
            if input_tokens == 0:
                input_tokens = sum(len(msg["content"].split()) for msg in messages)
            if output_tokens == 0 and content:
                output_tokens = len(content.split())
            if total_tokens == 0:
                total_tokens = input_tokens + output_tokens
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                },
                "raw_response": result
            }
    
    async def _call_openai(self, endpoint: str, api_key: str, model: str,
                          messages: List[Dict], options: Dict) -> Dict[str, Any]:
        """Call OpenAI API"""
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Add options with validation
        if "max_tokens" in options and options["max_tokens"] is not None:
            payload["max_tokens"] = options["max_tokens"]
        if "temperature" in options and options["temperature"] is not None:
            payload["temperature"] = max(0.0, min(2.0, options["temperature"]))
        if "top_p" in options and options["top_p"] is not None:
            payload["top_p"] = max(0.0, min(1.0, options["top_p"]))
        if "stop" in options and options["stop"] is not None:
            payload["stop"] = options["stop"]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error {response.status}: {error_text}")
            
            result = await response.json()
            
            content = ""
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
            
            usage = result.get("usage", {})
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                "raw_response": result
            }
    
    async def _call_anthropic(self, endpoint: str, api_key: str, model: str,
                             messages: List[Dict], options: Dict, version: str) -> Dict[str, Any]:
        """Call Anthropic API"""
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": options.get("max_tokens", 1024)
        }
        
        if system_message:
            payload["system"] = system_message
        
        # Add options with validation
        if "temperature" in options and options["temperature"] is not None:
            payload["temperature"] = max(0.0, min(1.0, options["temperature"]))
        if "top_p" in options and options["top_p"] is not None:
            payload["top_p"] = max(0.0, min(1.0, options["top_p"]))
        if "top_k" in options and options["top_k"] is not None:
            payload["top_k"] = max(1, options["top_k"])
        if "stop" in options and options["stop"] is not None:
            payload["stop_sequences"] = options["stop"]
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": version or "2023-06-01",
            "Content-Type": "application/json"
        }
        
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Anthropic API error {response.status}: {error_text}")
            
            result = await response.json()
            
            content = ""
            if "content" in result and result["content"]:
                content = result["content"][0].get("text", "")
            
            usage = result.get("usage", {})
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                },
                "raw_response": result
            }

# =============================================================================
# INTELLIGENT ROUTER - CORRECTED
# =============================================================================

class IntelligentRouter:
    """Core routing engine with intelligent combination selection - NOW SYNCHRONOUS"""
    
    def __init__(self, config_manager: ConfigManager, 
                 rate_limit_manager: RateLimitManager,
                 queue_manager: QueueManager,
                 llm_client: LLMClient):
        self.config = config_manager
        self.rate_limits = rate_limit_manager
        self.queues = queue_manager
        self.llm_client = llm_client
    
    async def route_intelligence_request(self, request: IntelligenceRequest) -> LLMResponse:
        """Route request based on intelligence level - WAITS FOR COMPLETION WITH RETRY LOGIC"""
        logger.info("Processing intelligence-based request", 
                   client_id=request.client_id,
                   intelligence_level=request.intelligence_level,
                   max_wait=request.max_wait_seconds)
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Get all models matching intelligence level
        model_combinations = self.config.get_models_by_intelligence(request.intelligence_level.value)
        
        logger.info("🔍 INTELLIGENCE DEBUG: Found models for intelligence level", 
                   intelligence_level=request.intelligence_level.value,
                   combinations=model_combinations,
                   count=len(model_combinations))
        
        if not model_combinations:
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="no_models_available",
                message=f"No models available for intelligence level: {request.intelligence_level.value}"
            )
        
        # Analyze all combinations (use max_wait_seconds for route calculation only)
        logger.info("🔍 ANALYSIS DEBUG: Starting analysis of all combinations")
        analyses = await self._analyze_all_combinations(model_combinations, request.max_wait_seconds)
        
        logger.info("🔍 ANALYSIS DEBUG: Analysis complete", 
                   valid_analyses=len(analyses),
                   total_combinations=len(model_combinations))
        
        if not analyses:
            return await self._create_failure_response(model_combinations, request.max_wait_seconds, "intelligence", request_id)
        
        # Sort by best combination (cheapest, then fastest)
        sorted_analyses = sorted(analyses, key=lambda x: (x.cost_per_1k_tokens, x.total_seconds))
        
        logger.info("🔍 SORTED DEBUG: All alternatives in order", 
                   alternatives_count=len(sorted_analyses),
                   alternatives=[{
                       "rank": i+1,
                       "provider": analysis.provider,
                       "model": analysis.model,
                       "api_key_id": analysis.api_key_id,
                       "cost": analysis.cost_per_1k_tokens,
                       "total_seconds": analysis.total_seconds
                   } for i, analysis in enumerate(sorted_analyses)])
        
        logger.info("🚀 RETRY DEBUG: Attempting request with retry logic", 
                   alternatives_count=len(sorted_analyses))
        
        # Try each alternative with retry logic - WITH EXCEPTION SAFETY
        try:
            result, used_analysis, attempt_info = await self._execute_with_retry_and_fallback(
                sorted_analyses, request.messages, request.options
            )
        except Exception as e:
            logger.error("💥 FATAL ERROR in retry logic", 
                        error=str(e),
                        traceback=traceback.format_exc())
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="retry_logic_failure",
                message=f"Fatal error in retry logic: {str(e)}"
            )
        
        total_time = time.time() - start_time
        
        if result["success"]:
            return LLMResponse(
                success=True,
                request_id=request_id,
                provider_used=used_analysis.provider,
                model_used=used_analysis.model,
                api_key_used=used_analysis.api_key_id,
                routing_method=RoutingMethod.INTELLIGENCE,
                decision_reason=f"cheapest_within_estimate_after_{attempt_info['total_attempts']}_attempts",
                response=result["response"],
                metadata=self._create_metadata_with_attempts(used_analysis, sorted_analyses, total_time, attempt_info)
            )
        else:
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="all_alternatives_failed",
                message=f"All {len(sorted_analyses)} alternatives failed after retry attempts. Last error: {result.get('error', 'Unknown error')}",
                evaluated_combinations=self._create_failure_summary(sorted_analyses, attempt_info)
            )
    
    async def route_scenario_request(self, request: ScenarioRequest) -> LLMResponse:
        """Route request based on scenario - WAITS FOR COMPLETION WITH RETRY LOGIC"""
        logger.info("Processing scenario-based request",
                   client_id=request.client_id,
                   scenario=request.scenario,
                   max_wait=request.max_wait_seconds)
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Get scenario combinations
        scenario_combinations = self.config.get_scenario_combinations(request.scenario)
        
        if not scenario_combinations:
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="scenario_not_found",
                message=f"Scenario not found: {request.scenario}"
            )
        
        # Convert to (provider, model) tuples and analyze
        model_combinations = [(provider, model) for provider, model, _ in scenario_combinations]
        analyses = await self._analyze_all_combinations(model_combinations, request.max_wait_seconds)
        
        if not analyses:
            return await self._create_failure_response(model_combinations, request.max_wait_seconds, "scenario", request_id)
        
        # Apply scenario priorities and sort
        priority_map = {(provider, model): priority for provider, model, priority in scenario_combinations}
        
        # Sort by priority first, then cost, then time
        sorted_analyses = sorted(analyses, key=lambda x: (
            priority_map.get((x.provider, x.model), 999),
            x.cost_per_1k_tokens,
            x.total_seconds
        ))
        
        logger.info("Attempting scenario request with retry logic", 
                   scenario=request.scenario,
                   alternatives_count=len(sorted_analyses))
        
        # Try each alternative with retry logic
        result, used_analysis, attempt_info = await self._execute_with_retry_and_fallback(
            sorted_analyses, request.messages, request.options
        )
        
        total_time = time.time() - start_time
        
        if result["success"]:
            return LLMResponse(
                success=True,
                request_id=request_id,
                provider_used=used_analysis.provider,
                model_used=used_analysis.model,
                api_key_used=used_analysis.api_key_id,
                routing_method=RoutingMethod.SCENARIO,
                decision_reason=f"scenario_priority_cost_optimized_after_{attempt_info['total_attempts']}_attempts",
                response=result["response"],
                metadata=self._create_metadata_with_attempts(used_analysis, analyses, total_time, attempt_info)
            )
        else:
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="all_alternatives_failed",
                message=f"All {len(sorted_analyses)} scenario alternatives failed after retry attempts. Last error: {result.get('error', 'Unknown error')}",
                evaluated_combinations=self._create_failure_summary(sorted_analyses, attempt_info)
            )
    
    async def route_direct_request(self, request: DirectRequest) -> LLMResponse:
        """Route request to specific provider/model - WAITS FOR COMPLETION WITH RETRY LOGIC"""
        logger.info("Processing direct request",
                   client_id=request.client_id,
                   provider=request.provider,
                   model=request.model_name,
                   api_key_id=request.api_key_id)
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Validate provider and model
        if request.provider not in self.config.providers:
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="provider_not_found",
                message=f"Provider not found: {request.provider}"
            )
        
        provider_config = self.config.providers[request.provider]
        
        if request.model_name not in provider_config.models:
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="model_not_found",
                message=f"Model not found: {request.model_name}"
            )
        
        # Determine API key(s) to use
        api_keys_to_check = []
        if request.api_key_id:
            if request.api_key_id in provider_config.keys:
                api_keys_to_check = [request.api_key_id]
            else:
                return LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="api_key_not_found",
                    message=f"API key not found: {request.api_key_id}"
                )
        else:
            # Auto-select from all available keys
            api_keys_to_check = list(provider_config.keys.keys())
        
        # Analyze combinations
        model_combinations = [(request.provider, request.model_name)]
        analyses = await self._analyze_combinations_with_keys(
            model_combinations, api_keys_to_check, request.max_wait_seconds
        )
        
        if not analyses:
            return await self._create_failure_response(
                model_combinations, request.max_wait_seconds, "direct", request_id
            )
        
        # Sort by fastest completion time for direct requests
        sorted_analyses = sorted(analyses, key=lambda x: x.total_seconds)
        
        logger.info("Attempting direct request with retry logic", 
                   provider=request.provider,
                   model=request.model_name,
                   alternatives_count=len(sorted_analyses))
        
        # Try each alternative with retry logic
        result, used_analysis, attempt_info = await self._execute_with_retry_and_fallback(
            sorted_analyses, request.messages, request.options
        )
        
        total_time = time.time() - start_time
        
        if result["success"]:
            return LLMResponse(
                success=True,
                request_id=request_id,
                provider_used=used_analysis.provider,
                model_used=used_analysis.model,
                api_key_used=used_analysis.api_key_id,
                routing_method=RoutingMethod.DIRECT,
                decision_reason=f"direct_fastest_completion_after_{attempt_info['total_attempts']}_attempts",
                response=result["response"],
                metadata=self._create_metadata_with_attempts(used_analysis, analyses, total_time, attempt_info)
            )
        else:
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="all_alternatives_failed",
                message=f"All {len(sorted_analyses)} direct alternatives failed after retry attempts. Last error: {result.get('error', 'Unknown error')}",
                evaluated_combinations=self._create_failure_summary(sorted_analyses, attempt_info)
            )
    
    async def _execute_with_retry_and_fallback(self, sorted_analyses: List[CompletionTimeAnalysis],
                                             messages: List[Message], options: Optional[LLMOptions]) -> Tuple[Dict[str, Any], CompletionTimeAnalysis, Dict[str, Any]]:
        """Execute request with retry logic and automatic failover - enhanced error handling"""
        MAX_RETRIES = 3
        attempt_info = {
            "alternatives_tried": 0,
            "total_attempts": 0,
            "failures": []
        }
        
        for analysis in sorted_analyses:
            attempt_info["alternatives_tried"] += 1
            
            logger.info("Trying alternative", 
                       alternative_num=attempt_info["alternatives_tried"],
                       provider=analysis.provider,
                       model=analysis.model,
                       api_key_id=analysis.api_key_id)
            
            # Try this alternative up to MAX_RETRIES times
            for retry_attempt in range(1, MAX_RETRIES + 1):
                attempt_info["total_attempts"] += 1
                
                logger.info("Attempt", 
                           alternative=attempt_info["alternatives_tried"],
                           retry=retry_attempt,
                           total_attempts=attempt_info["total_attempts"],
                           provider=analysis.provider,
                           model=analysis.model)
                
                try:
                    result = await self._execute_request(analysis, messages, options)
                    
                    if result["success"]:
                        logger.info("SUCCESS", 
                                   alternative=attempt_info["alternatives_tried"],
                                   retry=retry_attempt,
                                   provider=analysis.provider,
                                   model=analysis.model)
                        return result, analysis, attempt_info
                    else:
                        # Check if this is a failure that should skip retries and move to next provider
                        error_msg = result.get("error", "Unknown error")
                        should_skip_retries = self._should_skip_retries(error_msg)
                        
                        logger.warning("Attempt failed", 
                                     alternative=attempt_info["alternatives_tried"],
                                     retry=retry_attempt,
                                     provider=analysis.provider,
                                     model=analysis.model,
                                     error=error_msg,
                                     skip_retries=should_skip_retries)
                        
                        attempt_info["failures"].append({
                            "alternative": attempt_info["alternatives_tried"],
                            "retry": retry_attempt,
                            "provider": analysis.provider,
                            "model": analysis.model,
                            "api_key_id": analysis.api_key_id,
                            "error": error_msg,
                            "skip_retries": should_skip_retries
                        })
                        
                        # If should skip retries or this was the last retry, break to try next alternative
                        if should_skip_retries or retry_attempt == MAX_RETRIES:
                            if should_skip_retries:
                                logger.info("Skipping retries due to error type, moving to next provider", 
                                           provider=analysis.provider,
                                           model=analysis.model,
                                           error=error_msg)
                            else:
                                logger.error("Alternative exhausted after retries", 
                                           provider=analysis.provider,
                                           model=analysis.model,
                                           api_key_id=analysis.api_key_id,
                                           retries=MAX_RETRIES)
                            break
                        else:
                            # Wait a bit before retry (exponential backoff)
                            wait_time = min(2 ** (retry_attempt - 1), 5)  # 1s, 2s, 4s max
                            logger.info("Retrying after wait", 
                                       wait_seconds=wait_time,
                                       next_retry=retry_attempt + 1)
                            await asyncio.sleep(wait_time)
                            
                except Exception as e:
                    # Unexpected error (not from API response)
                    error_msg = f"Unexpected error: {str(e)}"
                    should_skip_retries = self._should_skip_retries(str(e))
                    
                    logger.error("Unexpected error during attempt", 
                               alternative=attempt_info["alternatives_tried"],
                               retry=retry_attempt,
                               provider=analysis.provider,
                               model=analysis.model,
                               error=error_msg,
                               skip_retries=should_skip_retries,
                               traceback=traceback.format_exc())
                    
                    attempt_info["failures"].append({
                        "alternative": attempt_info["alternatives_tried"],
                        "retry": retry_attempt,
                        "provider": analysis.provider,
                        "model": analysis.model,
                        "api_key_id": analysis.api_key_id,
                        "error": error_msg,
                        "skip_retries": should_skip_retries
                    })
                    
                    # For rate limiting or critical errors, skip retries
                    if should_skip_retries or retry_attempt == MAX_RETRIES:
                        break
                    else:
                        wait_time = min(2 ** (retry_attempt - 1), 5)
                        await asyncio.sleep(wait_time)
        
        # All alternatives exhausted
        last_error = attempt_info["failures"][-1]["error"] if attempt_info["failures"] else "No alternatives available"
        logger.error("All alternatives exhausted", 
                   alternatives_tried=attempt_info["alternatives_tried"],
                   total_attempts=attempt_info["total_attempts"],
                   last_error=last_error)
        
        return {
            "success": False,
            "error": f"All {attempt_info['alternatives_tried']} alternatives failed after {attempt_info['total_attempts']} total attempts"
        }, sorted_analyses[0] if sorted_analyses else None, attempt_info
    
    def _should_skip_retries(self, error_msg: str) -> bool:
        """Determine if an error should skip retries and move to next provider"""
        error_lower = error_msg.lower()
        
        # Rate limiting errors - skip retries
        if any(phrase in error_lower for phrase in [
            "rate limited", "429", "quota", "exceeded", "resource_exhausted"
        ]):
            return True
            
        # Authentication errors - skip retries  
        if any(phrase in error_lower for phrase in [
            "401", "403", "unauthorized", "forbidden", "invalid api key"
        ]):
            return True
            
        # Content/response errors that won't be fixed by retrying
        if any(phrase in error_lower for phrase in [
            "empty content", "could not extract content", "max_tokens"
        ]):
            return True
            
        # Network errors that might be temporary - allow retries
        if any(phrase in error_lower for phrase in [
            "timeout", "connection", "network", "500", "502", "503", "504"
        ]):
            return False
            
        # Default: allow retries for unknown errors
        return False
    
    async def _execute_request(self, analysis: CompletionTimeAnalysis, 
                             messages: List[Message], options: Optional[LLMOptions]) -> Dict[str, Any]:
        """Execute request to selected combination and wait for response - NO TIMEOUT"""
        provider_config = self.config.providers[analysis.provider]
        key_config = provider_config.keys[analysis.api_key_id]
        model_config = provider_config.models[analysis.model]
        
        request_data = {
            "provider": analysis.provider,
            "model": analysis.model,
            "api_key": key_config.api_key,
            "endpoint": model_config.endpoint,
            "version": getattr(model_config, 'version', None),
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "options": options.dict() if options else {}
        }
        
        # Execute directly and wait for response (no timeout - requests complete regardless of time)
        result = await self.queues.execute_request_directly(
            analysis.provider, analysis.model, analysis.api_key_id, request_data, 0,  # max_wait_seconds=0 (ignored)
            self.rate_limits, self.config, self.llm_client
        )
        
        return result
    
    def _create_metadata_with_attempts(self, selected: CompletionTimeAnalysis, 
                                     all_analyses: List[CompletionTimeAnalysis], 
                                     actual_time: float, attempt_info: Dict[str, Any]) -> ResponseMetadata:
        """Create response metadata including retry attempt information"""
        alternatives = []
        for analysis in all_analyses:
            if analysis.combination_key != selected.combination_key:
                alternatives.append({
                    "provider": analysis.provider,
                    "model": analysis.model,
                    "api_key": analysis.api_key_id,
                    "estimated_total_time_ms": int(analysis.total_seconds * 1000),
                    "cost_usd": analysis.cost_per_1k_tokens,
                    "reason_not_chosen": "higher_cost_or_longer_time_or_failed_retry"
                })
        
        return ResponseMetadata(
            rate_limit_wait_ms=int(selected.rate_limit_wait_seconds * 1000),
            queue_wait_ms=int(selected.queue_wait_seconds * 1000),
            api_response_time_ms=int(selected.processing_time_seconds * 1000),
            total_completion_time_ms=int(actual_time * 1000),
            cost_usd=selected.cost_per_1k_tokens,
            alternatives_considered=alternatives
        )
    
    def _create_failure_summary(self, sorted_analyses: List[CompletionTimeAnalysis], 
                              attempt_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed failure summary for debugging"""
        summary = []
        
        for i, analysis in enumerate(sorted_analyses, 1):
            # Find failures for this alternative
            alternative_failures = [f for f in attempt_info["failures"] if f["alternative"] == i]
            
            summary.append({
                "alternative_rank": i,
                "provider": analysis.provider,
                "model": analysis.model,
                "api_key": analysis.api_key_id,
                "estimated_total_time_seconds": analysis.total_seconds,
                "cost_per_1k_tokens": analysis.cost_per_1k_tokens,
                "retry_attempts": len(alternative_failures),
                "failures": alternative_failures
            })
        
        return summary
    
    async def _analyze_all_combinations(self, model_combinations: List[Tuple[str, str]], 
                                      max_wait_seconds: int) -> List[CompletionTimeAnalysis]:
        """Analyze all (provider, model, api_key) combinations for routing calculations"""
        analyses = []
        
        logger.info("🔍 ANALYSIS DEBUG: Starting analysis of combinations", 
                   total_combinations=len(model_combinations),
                   combinations=model_combinations,
                   max_wait_seconds=max_wait_seconds)
        
        for provider, model in model_combinations:
            logger.info("🔍 PROVIDER DEBUG: Analyzing provider/model", 
                       provider=provider, model=model)
            
            if provider not in self.config.providers:
                logger.warning("❌ PROVIDER DEBUG: Provider not found in config", 
                             provider=provider,
                             available_providers=list(self.config.providers.keys()))
                continue
            
            provider_config = self.config.providers[provider]
            
            if model not in provider_config.models:
                logger.warning("❌ MODEL DEBUG: Model not found in provider config", 
                             provider=provider, model=model,
                             available_models=list(provider_config.models.keys()))
                continue
            
            logger.info("✅ PROVIDER DEBUG: Provider and model found, checking API keys",
                       provider=provider, model=model,
                       available_keys=list(provider_config.keys.keys()))
            
            # Check all API keys for this provider/model combination
            key_count = 0
            for api_key_id, key_config in provider_config.keys.items():
                key_count += 1
                logger.info("🔑 KEY DEBUG: Checking API key", 
                           provider=provider, model=model, api_key_id=api_key_id,
                           key_num=f"{key_count}/{len(provider_config.keys)}")
                
                # Check if this key supports this model
                if model not in key_config.rate_limits:
                    logger.warning("❌ KEY DEBUG: Model not in rate limits for key", 
                                 provider=provider, model=model, api_key_id=api_key_id,
                                 available_models=list(key_config.rate_limits.keys()))
                    continue
                
                logger.info("✅ KEY DEBUG: Key supports model, performing analysis",
                           provider=provider, model=model, api_key_id=api_key_id)
                
                analysis = await self._analyze_combination(
                    provider, model, api_key_id, max_wait_seconds
                )
                
                if analysis:
                    logger.info("✅ ANALYSIS DEBUG: Analysis completed successfully", 
                               provider=provider, model=model, api_key_id=api_key_id,
                               total_seconds=analysis.total_seconds,
                               cost=analysis.cost_per_1k_tokens,
                               within_estimate=analysis.total_seconds <= max_wait_seconds)
                    
                    # Include all analyses (remove time filter for routing)
                    analyses.append(analysis)
                    logger.info("✅ ADDED DEBUG: Analysis added to alternatives",
                               provider=provider, model=model, api_key_id=api_key_id,
                               total_valid_alternatives=len(analyses))
                else:
                    logger.error("❌ ANALYSIS DEBUG: Analysis failed", 
                               provider=provider, model=model, api_key_id=api_key_id)
        
        logger.info("🏁 ANALYSIS DEBUG: Analysis complete", 
                   total_combinations_checked=len(model_combinations),
                   valid_combinations=len(analyses),
                   alternatives=[{
                       "provider": a.provider,
                       "model": a.model, 
                       "api_key_id": a.api_key_id,
                       "cost": a.cost_per_1k_tokens,
                       "total_seconds": a.total_seconds
                   } for a in analyses])
        
        return analyses
    
    async def _analyze_combinations_with_keys(self, model_combinations: List[Tuple[str, str]],
                                            api_key_ids: List[str], 
                                            max_wait_seconds: int) -> List[CompletionTimeAnalysis]:
        """Analyze specific (provider, model, api_key) combinations"""
        analyses = []
        
        for provider, model in model_combinations:
            if provider not in self.config.providers:
                continue
            
            provider_config = self.config.providers[provider]
            
            if model not in provider_config.models:
                continue
            
            for api_key_id in api_key_ids:
                if api_key_id not in provider_config.keys:
                    continue
                
                key_config = provider_config.keys[api_key_id]
                
                # Check if this key supports this model
                if model not in key_config.rate_limits:
                    continue
                
                analysis = await self._analyze_combination(
                    provider, model, api_key_id, max_wait_seconds
                )
                
                if analysis:
                    analyses.append(analysis)
        
        return analyses
    
    async def _analyze_combination(self, provider: str, model: str, api_key_id: str,
                                 max_wait_seconds: int) -> Optional[CompletionTimeAnalysis]:
        """Analyze a specific (provider, model, api_key) combination"""
        try:
            provider_config = self.config.providers[provider]
            key_config = provider_config.keys[api_key_id]
            model_config = provider_config.models[model]
            
            # Get rate limits for this combination
            rate_limits = key_config.rate_limits[model]
            rpm_limit = rate_limits["rpm"]
            rpd_limit = rate_limits["rpd"]
            
            # Check rate limit status
            is_limited, rate_wait = await self.rate_limits.check_rate_limit_status(
                provider, model, api_key_id, rpm_limit, rpd_limit
            )
            
            # Get queue wait time
            queue_wait = await self.queues.get_queue_wait_time(
                provider, model, api_key_id, model_config.avg_response_time_ms
            )
            
            # Calculate total completion time
            processing_time = model_config.avg_response_time_ms / 1000  # Convert to seconds
            total_time = rate_wait + queue_wait + processing_time
            
            combination_key = f"{provider}_{api_key_id}_{model}"
            
            return CompletionTimeAnalysis(
                rate_limit_wait_seconds=rate_wait,
                queue_wait_seconds=queue_wait,
                processing_time_seconds=processing_time,
                total_seconds=total_time,
                cost_per_1k_tokens=model_config.cost_per_1k_tokens,
                combination_key=combination_key,
                provider=provider,
                model=model,
                api_key_id=api_key_id
            )
            
        except Exception as e:
            logger.error("Error analyzing combination", 
                        provider=provider, model=model, api_key_id=api_key_id, error=str(e))
            return None
    
    async def _create_failure_response(self, model_combinations: List[Tuple[str, str]],
                                     max_wait_seconds: int, routing_method: str, request_id: str) -> LLMResponse:
        """Create failure response with suggestions"""
        # Find fastest available combination (ignoring time limit)
        all_analyses = []
        for provider, model in model_combinations:
            if provider in self.config.providers:
                provider_config = self.config.providers[provider]
                if model in provider_config.models:
                    for api_key_id in provider_config.keys:
                        analysis = await self._analyze_combination(provider, model, api_key_id, 999999)
                        if analysis:
                            all_analyses.append(analysis)
        
        fastest = min(all_analyses, key=lambda x: x.total_seconds) if all_analyses else None
        
        retry_suggestion = None
        if fastest:
            available_at = datetime.now(timezone.utc) + timedelta(seconds=fastest.total_seconds)
            retry_suggestion = {
                "fastest_available_combination": {
                    "provider": fastest.provider,
                    "model": fastest.model,
                    "api_key": fastest.api_key_id,
                    "estimated_total_time_seconds": fastest.total_seconds,
                    "available_at": available_at.isoformat()
                }
            }
        
        evaluated_combinations = []
        for analysis in all_analyses:
            evaluated_combinations.append({
                "provider": analysis.provider,
                "model": analysis.model,
                "api_key": analysis.api_key_id,
                "total_estimated_time_seconds": analysis.total_seconds,
                "breakdown": {
                    "rate_limit_wait": analysis.rate_limit_wait_seconds,
                    "queue_wait": analysis.queue_wait_seconds,
                    "processing_time": analysis.processing_time_seconds
                }
            })
        
        return LLMResponse(
            success=False,
            request_id=request_id,
            error_code="no_combinations_available",
            message=f"No available combinations for {routing_method} routing",
            retry_suggestion=retry_suggestion,
            evaluated_combinations=evaluated_combinations
        )

# =============================================================================
# GLOBAL MANAGERS - INITIALIZE IN PROPER ORDER
# =============================================================================

# Initialize managers but don't create router yet
config_manager = None
rate_limit_manager = None
queue_manager = None
llm_client = None
router = None

# =============================================================================
# LIFESPAN CONTEXT MANAGER - CORRECTED
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global config_manager, rate_limit_manager, queue_manager, llm_client, router
    
    # Startup
    logger.info("Starting bkvy application")
    
    try:
        # Initialize managers in proper order
        config_manager = ConfigManager()
        rate_limit_manager = RateLimitManager()
        queue_manager = QueueManager()
        llm_client = LLMClient()
        
        # Load configurations
        await config_manager.load_configs()
        
        # Start LLM client
        await llm_client.start()
        
        # Initialize router with all dependencies
        router = IntelligentRouter(config_manager, rate_limit_manager, queue_manager, llm_client)
        
        logger.info("bkvy application started successfully")
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down bkvy application")
    
    try:
        if llm_client:
            await llm_client.stop()
        logger.info("bkvy application shutdown complete")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Advanced Multi-Tier LLM API Router",
    description="Intelligent routing system with cost-time-rate optimization - SYNCHRONOUS RESPONSES",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CORE ROUTING ENDPOINTS - SYNCHRONOUS
# =============================================================================

@app.post("/llm/intelligence", response_model=LLMResponse)
async def route_by_intelligence(request: IntelligenceRequest, 
                               background_tasks: BackgroundTasks):
    """Route request based on intelligence level (low/medium/high) - WAITS FOR RESPONSE"""
    background_tasks.add_task(config_manager.refresh_if_changed)
    return await router.route_intelligence_request(request)

@app.post("/llm/scenario", response_model=LLMResponse)
async def route_by_scenario(request: ScenarioRequest,
                           background_tasks: BackgroundTasks):
    """Route request based on predefined scenario - WAITS FOR RESPONSE"""
    background_tasks.add_task(config_manager.refresh_if_changed)
    return await router.route_scenario_request(request)

@app.post("/llm/direct", response_model=LLMResponse)
async def route_direct(request: DirectRequest,
                      background_tasks: BackgroundTasks):
    """Route request to specific provider/model/key combination - WAITS FOR RESPONSE"""
    background_tasks.add_task(config_manager.refresh_if_changed)
    return await router.route_direct_request(request)

# =============================================================================
# STATUS AND MONITORING ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "mode": "synchronous"
    }

@app.get("/intelligence/{level}")
async def get_intelligence_models(level: IntelligenceLevel):
    """Get available models for intelligence level with current status"""
    await config_manager.refresh_if_changed()
    
    combinations = config_manager.get_models_by_intelligence(level.value)
    models_info = []
    
    for provider, model in combinations:
        provider_config = config_manager.providers[provider]
        model_config = provider_config.models[model]
        
        # Get status for each key
        keys_status = []
        for api_key_id, key_config in provider_config.keys.items():
            if model in key_config.rate_limits:
                rate_limits = key_config.rate_limits[model]
                is_limited, wait_time = await rate_limit_manager.check_rate_limit_status(
                    provider, model, api_key_id, rate_limits["rpm"], rate_limits["rpd"]
                )
                queue_wait = await queue_manager.get_queue_wait_time(
                    provider, model, api_key_id, model_config.avg_response_time_ms
                )
                
                keys_status.append({
                    "api_key_id": api_key_id,
                    "rate_limited": is_limited,
                    "rate_limit_wait_seconds": wait_time,
                    "queue_wait_seconds": queue_wait,
                    "total_wait_seconds": wait_time + queue_wait
                })
        
        models_info.append({
            "provider": provider,
            "model": model,
            "intelligence_tier": model_config.intelligence_tier,
            "cost_per_1k_tokens": model_config.cost_per_1k_tokens,
            "avg_response_time_ms": model_config.avg_response_time_ms,
            "keys_status": keys_status
        })
    
    return {
        "intelligence_level": level.value,
        "models": models_info
    }

@app.get("/scenarios")
async def get_scenarios():
    """Get all available scenarios"""
    await config_manager.refresh_if_changed()
    return {"scenarios": config_manager.scenarios}

@app.get("/providers") 
async def get_providers():
    """Get all providers with current status"""
    await config_manager.refresh_if_changed()
    
    providers_info = {}
    for provider_name, config in config_manager.providers.items():
        models_info = {}
        for model_name, model in config.models.items():
            keys_info = {}
            for key_id, key in config.keys.items():
                if model_name in key.rate_limits:
                    rate_limits = key.rate_limits[model_name]
                    is_limited, wait_time = await rate_limit_manager.check_rate_limit_status(
                        provider_name, model_name, key_id, rate_limits["rpm"], rate_limits["rpd"]
                    )
                    queue_wait = await queue_manager.get_queue_wait_time(
                        provider_name, model_name, key_id, model.avg_response_time_ms
                    )
                    
                    keys_info[key_id] = {
                        "rate_limits": rate_limits,
                        "currently_limited": is_limited,
                        "wait_time_seconds": wait_time,
                        "queue_wait_seconds": queue_wait
                    }
            
            models_info[model_name] = {
                "intelligence_tier": model.intelligence_tier,
                "cost_per_1k_tokens": model.cost_per_1k_tokens,
                "avg_response_time_ms": model.avg_response_time_ms,
                "keys": keys_info
            }
        
        providers_info[provider_name] = {
            "models": models_info
        }
    
    return {"providers": providers_info}

@app.get("/queue/status")
async def get_queue_status():
    """Get current queue states"""
    queue_states = await queue_manager.get_all_states()
    return {"queue_states": queue_states}

@app.get("/debug/config")
async def debug_config():
    """Debug endpoint to see configuration"""
    await config_manager.refresh_if_changed()
    
    debug_info = {}
    
    for provider_name, provider_config in config_manager.providers.items():
        provider_info = {
            "models": {},
            "keys": list(provider_config.keys.keys())
        }
        
        for model_name, model_config in provider_config.models.items():
            provider_info["models"][model_name] = {
                "intelligence_tier": model_config.intelligence_tier,
                "cost_per_1k_tokens": model_config.cost_per_1k_tokens,
                "avg_response_time_ms": model_config.avg_response_time_ms
            }
        
        debug_info[provider_name] = provider_info
    
    # Also show intelligence level mappings
    intelligence_mappings = {}
    for level in ["low", "medium", "high"]:
        combinations = config_manager.get_models_by_intelligence(level)
        intelligence_mappings[level] = combinations
    
    return {
        "providers": debug_info,
        "intelligence_mappings": intelligence_mappings,
        "scenarios": config_manager.scenarios
    }

@app.get("/rates/status")
async def get_rate_status():
    """Get current rate limit states"""
    rate_states = await rate_limit_manager.get_all_states()
    return {"rate_limit_states": rate_states}

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Setup logging directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 10006))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    workers = int(os.getenv("WORKERS", 1))
    
    # Run the server
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=workers,
        reload=False
    )
