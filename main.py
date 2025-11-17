#!/usr/bin/env python3
"""
Main entry point for bkvy application

A sophisticated HTTP server that manages complex rate limiting across multiple API keys 
and models, with intelligent routing that dynamically selects optimal providers based 
on real-time queue states, rate limit statuses, costs, and time constraints.
"""

import os
import uvicorn
from pathlib import Path

from bkvy.utils.logging import setup_logging
from bkvy.api.app import create_app

# Setup logging
logger = setup_logging()


def main():
    """Main entry point for the application"""
    
    # Setup logging directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 10006))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    workers = int(os.getenv("WORKERS", 1))
    
    logger.info("Starting bkvy", 
               host=host, port=port, log_level=log_level, workers=workers)
    
    # Create the FastAPI application
    app = create_app()
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        workers=workers,
        reload=False
    )


if __name__ == "__main__":
    main()