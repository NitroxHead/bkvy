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
from ..utils.logging import setup_logging

logger = setup_logging()

# Global managers - initialized during lifespan
config_manager = None
rate_limit_manager = None
queue_manager = None
llm_client = None
router = None


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