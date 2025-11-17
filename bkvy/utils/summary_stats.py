"""
Summary statistics logger - lightweight daily aggregates without detailed transaction data
"""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from .logging import setup_logging

logger = setup_logging()


class SummaryStatsLogger:
    """Lightweight statistics aggregator that maintains daily pivot tables"""

    def __init__(self, enabled: bool = True, log_dir: str = "logs"):
        """
        Initialize summary statistics logger

        Args:
            enabled: Whether summary stats logging is enabled
            log_dir: Directory for log files
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.stats_file = self.log_dir / "daily_stats.json"
        self._stats_lock = asyncio.Lock()

        # In-memory daily stats (pivot table structure)
        self._daily_stats = defaultdict(lambda: {
            "total_requests": 0,
            "successful": 0,
            "by_routing_method": defaultdict(int),
            "by_intelligence": defaultdict(int),
            "by_provider": defaultdict(int),
            "errors": defaultdict(int),
            "total_cost": 0.0,
            "total_time_ms": 0,
            "response_count": 0  # For average calculation
        })

        if self.enabled:
            self.log_dir.mkdir(exist_ok=True)
            self._load_stats()

    def _load_stats(self):
        """Load existing daily stats from JSON file"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert loaded data to defaultdict structure
                    for date, stats in data.items():
                        self._daily_stats[date] = {
                            "total_requests": stats.get("total_requests", 0),
                            "successful": stats.get("successful", 0),
                            "by_routing_method": defaultdict(int, stats.get("by_routing_method", {})),
                            "by_intelligence": defaultdict(int, stats.get("by_intelligence", {})),
                            "by_provider": defaultdict(int, stats.get("by_provider", {})),
                            "errors": defaultdict(int, stats.get("errors", {})),
                            "total_cost": stats.get("total_cost", 0.0),
                            "total_time_ms": stats.get("total_time_ms", 0),
                            "response_count": stats.get("response_count", 0)
                        }
                logger.info("Summary stats loaded", dates=len(self._daily_stats))
            except Exception as e:
                logger.error("Failed to load summary stats", error=str(e))

    async def _flush_stats(self):
        """Flush daily stats to JSON file"""
        try:
            async with self._stats_lock:
                # Convert defaultdicts to regular dicts for JSON serialization
                output = {}
                for date, stats in self._daily_stats.items():
                    output[date] = {
                        "total_requests": stats["total_requests"],
                        "successful": stats["successful"],
                        "by_routing_method": dict(stats["by_routing_method"]),
                        "by_intelligence": dict(stats["by_intelligence"]),
                        "by_provider": dict(stats["by_provider"]),
                        "errors": dict(stats["errors"]),
                        "total_cost": round(stats["total_cost"], 6),
                        "avg_response_time_ms": int(stats["total_time_ms"] / stats["response_count"]) if stats["response_count"] > 0 else 0
                    }

                with open(self.stats_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2)

                logger.debug("Summary stats flushed to disk")

        except Exception as e:
            logger.error("Failed to flush summary stats", error=str(e))

    async def log_request(self,
                         success: bool,
                         routing_method: str,
                         intelligence_level: Optional[str] = None,
                         provider_used: Optional[str] = None,
                         error_type: Optional[str] = None,
                         cost_estimate: Optional[float] = None,
                         total_time_ms: Optional[int] = None):
        """
        Log a request summary (minimal data, no detailed transaction info)

        Args:
            success: Whether request succeeded
            routing_method: intelligence/scenario/direct
            intelligence_level: low/medium/high (if applicable)
            provider_used: Provider that handled the request
            error_type: Error type if failed
            cost_estimate: Estimated cost in USD
            total_time_ms: Total processing time in milliseconds
        """
        if not self.enabled:
            return

        try:
            # Get current date
            date_key = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            stats = self._daily_stats[date_key]

            # Update counters
            stats["total_requests"] += 1
            if success:
                stats["successful"] += 1

            # Routing method
            if routing_method:
                stats["by_routing_method"][routing_method] += 1

            # Intelligence level
            if intelligence_level:
                stats["by_intelligence"][intelligence_level] += 1

            # Provider
            if provider_used:
                stats["by_provider"][provider_used] += 1

            # Errors
            if error_type:
                stats["errors"][error_type] += 1

            # Cost
            if cost_estimate:
                stats["total_cost"] += cost_estimate

            # Response time
            if total_time_ms:
                stats["total_time_ms"] += total_time_ms
                stats["response_count"] += 1

            # Flush to disk
            await self._flush_stats()

            logger.debug("Summary stats updated", date=date_key)

        except Exception as e:
            logger.error("Failed to log summary stats", error=str(e))

    async def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific date or all dates"""
        if not self.enabled:
            return {"enabled": False}

        try:
            if date:
                # Return specific date
                if date in self._daily_stats:
                    stats = self._daily_stats[date]
                    return {
                        "date": date,
                        "total_requests": stats["total_requests"],
                        "successful": stats["successful"],
                        "success_rate": stats["successful"] / stats["total_requests"] if stats["total_requests"] > 0 else 0,
                        "by_routing_method": dict(stats["by_routing_method"]),
                        "by_intelligence": dict(stats["by_intelligence"]),
                        "by_provider": dict(stats["by_provider"]),
                        "errors": dict(stats["errors"]),
                        "total_cost": round(stats["total_cost"], 6),
                        "avg_response_time_ms": int(stats["total_time_ms"] / stats["response_count"]) if stats["response_count"] > 0 else 0
                    }
                else:
                    return {"error": f"No data for date: {date}"}
            else:
                # Return all dates
                result = {}
                for date_key, stats in self._daily_stats.items():
                    result[date_key] = {
                        "total_requests": stats["total_requests"],
                        "successful": stats["successful"],
                        "success_rate": stats["successful"] / stats["total_requests"] if stats["total_requests"] > 0 else 0,
                        "by_routing_method": dict(stats["by_routing_method"]),
                        "by_intelligence": dict(stats["by_intelligence"]),
                        "by_provider": dict(stats["by_provider"]),
                        "errors": dict(stats["errors"]),
                        "total_cost": round(stats["total_cost"], 6),
                        "avg_response_time_ms": int(stats["total_time_ms"] / stats["response_count"]) if stats["response_count"] > 0 else 0
                    }
                return result

        except Exception as e:
            logger.error("Failed to get daily stats", error=str(e))
            return {"error": str(e)}

    async def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across all days"""
        if not self.enabled:
            return {"enabled": False}

        try:
            # Aggregate across all days
            total_requests = 0
            successful_requests = 0
            routing_methods = defaultdict(int)
            providers_used = defaultdict(int)
            intelligence_levels = defaultdict(int)
            errors = defaultdict(int)
            total_cost = 0.0

            for date, stats in self._daily_stats.items():
                total_requests += stats["total_requests"]
                successful_requests += stats["successful"]
                total_cost += stats["total_cost"]

                for method, count in stats["by_routing_method"].items():
                    routing_methods[method] += count

                for provider, count in stats["by_provider"].items():
                    providers_used[provider] += count

                for level, count in stats["by_intelligence"].items():
                    intelligence_levels[level] += count

                for error, count in stats["errors"].items():
                    errors[error] += count

            return {
                "enabled": True,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "routing_methods": dict(routing_methods),
                "providers_used": dict(providers_used),
                "intelligence_levels": dict(intelligence_levels),
                "errors": dict(errors),
                "total_cost_estimate": round(total_cost, 6),
                "days_tracked": len(self._daily_stats),
                "stats_file": str(self.stats_file)
            }

        except Exception as e:
            logger.error("Failed to generate aggregate stats", error=str(e))
            return {"enabled": True, "error": str(e)}


# Global instance that can be configured
_summary_stats_logger: Optional[SummaryStatsLogger] = None


def init_summary_stats_logger(enabled: bool = True, log_dir: str = "logs") -> SummaryStatsLogger:
    """Initialize the global summary stats logger"""
    global _summary_stats_logger
    _summary_stats_logger = SummaryStatsLogger(enabled=enabled, log_dir=log_dir)
    return _summary_stats_logger


def get_summary_stats_logger() -> Optional[SummaryStatsLogger]:
    """Get the global summary stats logger instance"""
    return _summary_stats_logger


def is_summary_stats_enabled() -> bool:
    """Check if summary stats logging is enabled"""
    return _summary_stats_logger is not None and _summary_stats_logger.enabled
