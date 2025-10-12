"""
Application lifespan management for FastAPI
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from ..core.config import ConfigManager
from ..core.rate_limits import RateLimitManager
from ..core.queues import QueueManager
from ..core.llm_client import LLMClient
from ..core.router import IntelligentRouter
from ..core.circuit_breaker import CircuitBreakerManager
from ..core.health_probe import HealthProbe, BackgroundProbeWorker
from ..utils.logging import setup_logging
from ..utils.transaction_logger import init_transaction_logger
from ..utils.summary_stats import init_summary_stats_logger
from ..utils.dashboard import init_dashboard_processor

logger = setup_logging()

# Global managers - initialized during lifespan
config_manager = None
rate_limit_manager = None
queue_manager = None
llm_client = None
router = None
dashboard_processor = None
circuit_breaker = None
health_probe = None
probe_worker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global config_manager, rate_limit_manager, queue_manager, llm_client, router, dashboard_processor
    global circuit_breaker, health_probe, probe_worker

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

        # Initialize logging (both systems can be independently enabled/disabled)
        import os
        log_dir = os.getenv("LOG_DIR", "logs")

        # Detailed transaction logging (CSV with full request details)
        transaction_logging_enabled = os.getenv("TRANSACTION_LOGGING", "false").lower() == "true"
        transaction_logger = init_transaction_logger(
            enabled=transaction_logging_enabled,
            log_dir=log_dir
        )
        logger.info("Transaction logging initialized",
                   enabled=transaction_logging_enabled,
                   log_dir=log_dir)

        # Summary statistics logging (JSON with daily aggregates)
        summary_stats_enabled = os.getenv("SUMMARY_STATS", "false").lower() == "true"
        summary_stats_logger = init_summary_stats_logger(
            enabled=summary_stats_enabled,
            log_dir=log_dir
        )
        logger.info("Summary stats logging initialized",
                   enabled=summary_stats_enabled,
                   log_dir=log_dir)

        # Initialize dashboard processor
        dashboard_enabled = os.getenv("DASHBOARD_ENABLED", "false").lower() == "true"
        if dashboard_enabled:
            dashboard_processor = init_dashboard_processor(log_dir=log_dir)
            dashboard_ips = os.getenv("DASHBOARD_ALLOWED_IPS", "127.0.0.1")
            logger.info("Dashboard initialized",
                       enabled=dashboard_enabled,
                       allowed_ips=dashboard_ips,
                       log_dir=log_dir)
        else:
            logger.info("Dashboard disabled")

        # Start LLM client
        await llm_client.start()

        # Initialize circuit breaker
        circuit_breaker = CircuitBreakerManager(config_manager)
        await circuit_breaker.initialize()
        logger.info("Circuit breaker initialized",
                   enabled=circuit_breaker.enabled,
                   total_circuits=len(circuit_breaker.circuits))

        # Initialize health probe and background worker
        health_probe = HealthProbe(llm_client)
        probe_worker = BackgroundProbeWorker(circuit_breaker, config_manager, health_probe)
        await probe_worker.start()
        logger.info("Health probe worker initialized",
                   enabled=probe_worker.enabled,
                   interval_seconds=probe_worker.interval_seconds)

        # Initialize router with all dependencies
        router = IntelligentRouter(
            config_manager,
            rate_limit_manager,
            queue_manager,
            llm_client,
            circuit_breaker
        )

        logger.info("bkvy application started successfully")

    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down bkvy application")

    try:
        # Stop background probe worker
        if probe_worker:
            await probe_worker.stop()
            logger.info("Health probe worker stopped")

        # Stop LLM client
        if llm_client:
            await llm_client.stop()

        logger.info("bkvy application shutdown complete")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager"""
    return config_manager


def get_rate_limit_manager() -> RateLimitManager:
    """Get the global rate limit manager"""
    return rate_limit_manager


def get_queue_manager() -> QueueManager:
    """Get the global queue manager"""
    return queue_manager


def get_router() -> IntelligentRouter:
    """Get the global intelligent router"""
    return router


def get_circuit_breaker() -> CircuitBreakerManager:
    """Get the global circuit breaker manager"""
    return circuit_breaker


def get_health_probe() -> HealthProbe:
    """Get the global health probe"""
    return health_probe