"""
Logging configuration for bkvy
"""

import logging
import os

import structlog


def setup_logging():
    """Configure structured logging for the application"""
    # Set root logger level from LOG_LEVEL env var so filter_by_level works
    log_level = os.getenv("LOG_LEVEL", "info").upper()
    logging.basicConfig(format="%(message)s", level=getattr(logging, log_level, logging.INFO))

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

    return structlog.get_logger()