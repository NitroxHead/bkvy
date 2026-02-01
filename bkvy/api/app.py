"""
FastAPI application and endpoints for bkvy
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from ..models.enums import IntelligenceLevel
from ..models.schemas import (
    IntelligenceRequest, ScenarioRequest, DirectRequest, LLMResponse, SimplifiedResponse
)
from typing import Union
from .lifespan import (
    lifespan, get_config_manager, get_rate_limit_manager, get_queue_manager,
    get_router, get_circuit_breaker, get_health_probe
)
from .middleware import IPWhitelistMiddleware, parse_ip_list
from ..utils.transaction_logger import get_transaction_logger
from ..utils.summary_stats import get_summary_stats_logger
from ..utils.dashboard import get_dashboard_processor


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

    # Add IP whitelist middleware for dashboard (if enabled)
    dashboard_enabled = os.getenv("DASHBOARD_ENABLED", "false").lower() == "true"
    if dashboard_enabled:
        dashboard_ips = os.getenv("DASHBOARD_ALLOWED_IPS", "127.0.0.1")
        allowed_ips = parse_ip_list(dashboard_ips)
        app.add_middleware(
            IPWhitelistMiddleware,
            allowed_ips=allowed_ips,
            protected_paths=["/dashboard"]
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

    # =============================================================================
    # CIRCUIT BREAKER ENDPOINTS
    # =============================================================================

    @app.get("/circuits/status")
    async def get_circuit_status():
        """Get all circuit breaker states"""
        circuit_breaker = get_circuit_breaker()
        if not circuit_breaker or not circuit_breaker.enabled:
            return {
                "enabled": False,
                "message": "Circuit breaker is disabled"
            }

        return {
            "enabled": True,
            "circuits": circuit_breaker.get_all_circuit_states(),
            "total_circuits": len(circuit_breaker.circuits),
            "open_circuits": sum(1 for c in circuit_breaker.circuits.values() if c.state.value == "open"),
            "half_open_circuits": sum(1 for c in circuit_breaker.circuits.values() if c.state.value == "half_open"),
            "closed_circuits": sum(1 for c in circuit_breaker.circuits.values() if c.state.value == "closed")
        }

    @app.get("/circuits/provider/{provider_name}")
    async def get_provider_circuit_health(provider_name: str):
        """Get circuit breaker health for a specific provider"""
        circuit_breaker = get_circuit_breaker()
        if not circuit_breaker or not circuit_breaker.enabled:
            return {
                "enabled": False,
                "message": "Circuit breaker is disabled"
            }

        health = await circuit_breaker.get_provider_health(provider_name)
        return health.to_dict()

    @app.get("/circuits/{provider}/{model}/{api_key_id}")
    async def get_specific_circuit(provider: str, model: str, api_key_id: str):
        """Get specific circuit state"""
        circuit_breaker = get_circuit_breaker()
        if not circuit_breaker or not circuit_breaker.enabled:
            return {
                "enabled": False,
                "message": "Circuit breaker is disabled"
            }

        circuit = circuit_breaker.get_circuit_state(provider, model, api_key_id)
        if circuit:
            return circuit
        else:
            return {
                "error": "Circuit not found",
                "provider": provider,
                "model": model,
                "api_key_id": api_key_id
            }

    @app.post("/circuits/reset/{provider}/{model}/{api_key_id}")
    async def reset_circuit(provider: str, model: str, api_key_id: str):
        """Manually reset a circuit (admin action)"""
        circuit_breaker = get_circuit_breaker()
        if not circuit_breaker or not circuit_breaker.enabled:
            return {
                "enabled": False,
                "message": "Circuit breaker is disabled"
            }

        success = await circuit_breaker.reset_circuit(provider, model, api_key_id)
        if success:
            return {
                "success": True,
                "message": f"Circuit reset for {provider}/{model}/{api_key_id}"
            }
        else:
            return {
                "success": False,
                "message": "Circuit not found"
            }

    @app.get("/circuits/summary")
    async def get_circuits_summary():
        """Get summary of all providers' circuit health"""
        circuit_breaker = get_circuit_breaker()
        config_manager = get_config_manager()

        if not circuit_breaker or not circuit_breaker.enabled:
            return {
                "enabled": False,
                "message": "Circuit breaker is disabled"
            }

        summary = {}
        for provider_name in config_manager.providers.keys():
            health = await circuit_breaker.get_provider_health(provider_name)
            summary[provider_name] = health.to_dict()

        return {
            "enabled": True,
            "providers": summary
        }

    # =============================================================================
    # TRANSACTION STATISTICS ENDPOINTS
    # =============================================================================

    @app.get("/statistics/summary")
    async def get_statistics_summary():
        """Get transaction statistics summary"""
        transaction_logger = get_transaction_logger()
        if not transaction_logger:
            return {"enabled": False, "message": "Transaction logging is disabled"}

        stats = await transaction_logger.get_stats_summary()
        return stats

    @app.get("/statistics/status")
    async def get_statistics_status():
        """Get transaction logging status"""
        transaction_logger = get_transaction_logger()
        summary_logger = get_summary_stats_logger()

        return {
            "transaction_logging": {
                "enabled": transaction_logger.enabled if transaction_logger else False,
                "log_file": str(transaction_logger.csv_file) if transaction_logger else None,
                "log_directory": str(transaction_logger.log_dir) if transaction_logger else None
            },
            "summary_stats": {
                "enabled": summary_logger.enabled if summary_logger else False,
                "stats_file": str(summary_logger.stats_file) if summary_logger else None,
                "log_directory": str(summary_logger.log_dir) if summary_logger else None
            }
        }

    # =============================================================================
    # SUMMARY STATISTICS ENDPOINTS
    # =============================================================================

    @app.get("/statistics/daily/{date}")
    async def get_daily_statistics(date: str):
        """Get statistics for a specific date (YYYY-MM-DD format)"""
        summary_logger = get_summary_stats_logger()
        if not summary_logger:
            return {"enabled": False, "message": "Summary statistics is disabled"}

        stats = await summary_logger.get_daily_stats(date=date)
        return stats

    @app.get("/statistics/daily")
    async def get_all_daily_statistics():
        """Get statistics for all dates (daily pivot table)"""
        summary_logger = get_summary_stats_logger()
        if not summary_logger:
            return {"enabled": False, "message": "Summary statistics is disabled"}

        stats = await summary_logger.get_daily_stats()
        return stats

    @app.get("/statistics/aggregate")
    async def get_aggregate_statistics():
        """Get aggregated statistics across all days"""
        summary_logger = get_summary_stats_logger()
        if not summary_logger:
            return {"enabled": False, "message": "Summary statistics is disabled"}

        stats = await summary_logger.get_aggregate_stats()
        return stats

    # =============================================================================
    # DASHBOARD ENDPOINTS
    # =============================================================================

    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard():
        """Serve the dashboard HTML page (IP-restricted)"""
        dashboard_enabled = os.getenv("DASHBOARD_ENABLED", "false").lower() == "true"
        if not dashboard_enabled:
            return HTMLResponse(
                content="<h1>Dashboard Disabled</h1><p>Set DASHBOARD_ENABLED=true to enable the dashboard.</p>",
                status_code=503
            )

        # Read and serve the dashboard HTML template
        template_path = Path(__file__).parent / "templates" / "dashboard.html"

        if not template_path.exists():
            return HTMLResponse(
                content="<h1>Dashboard Not Found</h1><p>Dashboard template is missing.</p>",
                status_code=404
            )

        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        return HTMLResponse(content=html_content)

    @app.get("/dashboard/data")
    async def get_dashboard_data(
        hours: int = Query(24, ge=1, le=8760),
        start_date: str = Query(None, description="Start date (ISO format)"),
        end_date: str = Query(None, description="End date (ISO format)"),
        timezone_offset: int = Query(0, ge=-12, le=12, description="Timezone offset from UTC in hours")
    ):
        """Get dashboard data as JSON (IP-restricted)"""
        dashboard_enabled = os.getenv("DASHBOARD_ENABLED", "false").lower() == "true"
        if not dashboard_enabled:
            return {"error": "Dashboard is disabled"}

        dashboard_processor = get_dashboard_processor()
        if not dashboard_processor:
            return {"error": "Dashboard processor not initialized"}

        data = await dashboard_processor.get_dashboard_data(
            hours=hours,
            start_date=start_date,
            end_date=end_date,
            timezone_offset=timezone_offset
        )
        return data

    @app.get("/dashboard/health")
    async def get_dashboard_health():
        """Get dashboard system health information (IP-restricted)"""
        dashboard_enabled = os.getenv("DASHBOARD_ENABLED", "false").lower() == "true"
        if not dashboard_enabled:
            return {"enabled": False}

        dashboard_processor = get_dashboard_processor()
        if not dashboard_processor:
            return {"enabled": True, "error": "Dashboard processor not initialized"}

        health = await dashboard_processor.get_system_health()
        health["enabled"] = True
        return health

    return app