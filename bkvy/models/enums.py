"""
Enums for bkvy application
"""

from enum import Enum


class IntelligenceLevel(str, Enum):
    """Intelligence levels for LLM model classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RoutingMethod(str, Enum):
    """Routing methods supported by the router"""
    INTELLIGENCE = "intelligence"
    SCENARIO = "scenario"
    DIRECT = "direct"