"""
bkvy - Advanced Multi-Tier LLM API Router

A sophisticated HTTP server that manages complex rate limiting across multiple API keys 
and models, with intelligent routing that dynamically selects optimal providers based 
on real-time queue states, rate limit statuses, costs, and time constraints.
"""

__version__ = "1.0.0"
__author__ = "bkvy Development Team"
__description__ = "Advanced Multi-Tier LLM API Router with Intelligent Cost-Time-Rate Optimization"

# Import main components for easy access
from .core.config import ConfigManager
from .core.rate_limits import RateLimitManager  
from .core.queues import QueueManager
from .core.llm_client import LLMClient
from .core.router import IntelligentRouter

from .models.enums import IntelligenceLevel, RoutingMethod
from .models.schemas import (
    IntelligenceRequest, ScenarioRequest, DirectRequest, 
    LLMResponse, Message, LLMOptions
)

from .api.app import create_app
from .utils.transaction_logger import TransactionLogger, init_transaction_logger, get_transaction_logger
from .utils.summary_stats import SummaryStatsLogger, init_summary_stats_logger, get_summary_stats_logger

__all__ = [
    # Core managers
    'ConfigManager',
    'RateLimitManager',
    'QueueManager',
    'LLMClient',
    'IntelligentRouter',

    # Models and schemas
    'IntelligenceLevel',
    'RoutingMethod',
    'IntelligenceRequest',
    'ScenarioRequest',
    'DirectRequest',
    'LLMResponse',
    'Message',
    'LLMOptions',

    # Transaction logging (detailed CSV)
    'TransactionLogger',
    'init_transaction_logger',
    'get_transaction_logger',

    # Summary statistics (daily aggregates JSON)
    'SummaryStatsLogger',
    'init_summary_stats_logger',
    'get_summary_stats_logger',

    # App factory
    'create_app'
]