"""
FastAPI application and endpoints for bkvy
"""

from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from ..models.enums import IntelligenceLevel
from ..models.schemas import (
    IntelligenceRequest, ScenarioRequest, DirectRequest, LLMResponse, SimplifiedResponse
)
from typing import Union
from .lifespan import lifespan, get_config_manager, get_rate_limit_manager, get_queue_manager, get_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
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

    @app.post("/llm/intelligence", response_model=Union[LLMResponse, SimplifiedResponse])
    async def route_by_intelligence(request: IntelligenceRequest, 
                                   background_tasks: BackgroundTasks):
        """Route request based on intelligence level (low/medium/high) - WAITS FOR RESPONSE"""
        config_manager = get_config_manager()
        router = get_router()
        background_tasks.add_task(config_manager.refresh_if_changed)
        return await router.route_intelligence_request(request)

    @app.post("/llm/scenario", response_model=Union[LLMResponse, SimplifiedResponse])
    async def route_by_scenario(request: ScenarioRequest,
                               background_tasks: BackgroundTasks):
        """Route request based on predefined scenario - WAITS FOR RESPONSE"""
        config_manager = get_config_manager()
        router = get_router()
        background_tasks.add_task(config_manager.refresh_if_changed)
        return await router.route_scenario_request(request)

    @app.post("/llm/direct", response_model=Union[LLMResponse, SimplifiedResponse])
    async def route_direct(request: DirectRequest,
                          background_tasks: BackgroundTasks):
        """Route request to specific provider/model/key combination - WAITS FOR RESPONSE"""
        config_manager = get_config_manager()
        router = get_router()
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
        config_manager = get_config_manager()
        rate_limit_manager = get_rate_limit_manager()
        queue_manager = get_queue_manager()
        
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
        config_manager = get_config_manager()
        await config_manager.refresh_if_changed()
        return {"scenarios": config_manager.scenarios}

    @app.get("/providers") 
    async def get_providers():
        """Get all providers with current status"""
        config_manager = get_config_manager()
        rate_limit_manager = get_rate_limit_manager()
        queue_manager = get_queue_manager()
        
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
        queue_manager = get_queue_manager()
        queue_states = await queue_manager.get_all_states()
        return {"queue_states": queue_states}

    @app.get("/debug/config")
    async def debug_config():
        """Debug endpoint to see configuration"""
        config_manager = get_config_manager()
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
        rate_limit_manager = get_rate_limit_manager()
        rate_states = await rate_limit_manager.get_all_states()
        return {"rate_limit_states": rate_states}

    return app