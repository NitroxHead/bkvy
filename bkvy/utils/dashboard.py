"""
Dashboard data processing and statistics aggregation
"""

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .logging import setup_logging

logger = setup_logging()


class DashboardDataProcessor:
    """Process transaction logs and stats for dashboard visualization"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.transactions_file = self.log_dir / "transactions.csv"
        self.daily_stats_file = self.log_dir / "daily_stats.json"

    async def get_dashboard_data(self, hours: int = 24, start_date: str = None, end_date: str = None, timezone_offset: int = 0) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data

        Args:
            hours: Number of hours to include in recent data (default: 24)
            start_date: Start date in ISO format (optional, overrides hours)
            end_date: End date in ISO format (optional, defaults to now)
            timezone_offset: Timezone offset from UTC in hours (default: 0)

        Returns:
            Dictionary with all dashboard data
        """
        try:
            # Get transaction-based statistics
            transactions_data = await self._get_transaction_statistics(hours, start_date, end_date, timezone_offset)

            # Get daily stats
            daily_stats = await self._get_daily_statistics()

            # Combine and return
            return {
                "overview": transactions_data["overview"],
                "time_series": transactions_data["time_series"],
                "distributions": transactions_data["distributions"],
                "recent_transactions": transactions_data["recent_transactions"],
                "errors": transactions_data["errors"],
                "api_keys": transactions_data["api_keys"],
                "wait_time_analysis": transactions_data["wait_time_analysis"],
                "data_coverage": transactions_data["data_coverage"],
                "daily_summary": daily_stats,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error("Failed to generate dashboard data", error=str(e))
            return {"error": str(e)}

    async def _get_transaction_statistics(self, hours: int = 24, start_date: str = None, end_date: str = None, timezone_offset: int = 0) -> Dict[str, Any]:
        """Get statistics from transaction CSV logs"""
        if not self.transactions_file.exists():
            return self._empty_transaction_data()

        try:
            # Calculate time range (always work in UTC internally)
            if start_date:
                # Custom date range
                cutoff_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    end_time = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                else:
                    end_time = datetime.now(timezone.utc)
            else:
                # Hours-based range
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
                end_time = datetime.now(timezone.utc)

            # Metrics
            total_requests = 0
            successful_requests = 0
            total_cost = 0.0
            total_time_ms = 0
            time_count = 0

            # Response time tracking for percentiles
            response_times = []

            # Wait time compliance tracking
            wait_time_met = 0
            wait_time_exceeded = 0
            wait_time_total = 0

            # Distributions
            providers = defaultdict(int)
            models = defaultdict(int)
            routing_methods = defaultdict(int)
            intelligence_levels = defaultdict(int)
            clients = defaultdict(int)

            # Errors
            error_types = defaultdict(int)
            error_details = []

            # Time series (hourly buckets)
            time_series = defaultdict(lambda: {"success": 0, "failed": 0, "cost": 0.0})

            # Recent transactions
            recent_transactions = []

            # API Key usage tracking
            api_key_stats = defaultdict(lambda: {
                "total_requests": 0,
                "successful": 0,
                "failed": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "models_used": defaultdict(int),
                "avg_response_time": 0,
                "response_time_sum": 0
            })

            # Wait time analysis
            wait_time_buckets = {
                "0-10s": {"count": 0, "success": 0, "failed": 0},
                "10-30s": {"count": 0, "success": 0, "failed": 0},
                "30-60s": {"count": 0, "success": 0, "failed": 0},
                "60-120s": {"count": 0, "success": 0, "failed": 0},
                "120s+": {"count": 0, "success": 0, "failed": 0}
            }

            # Track actual data range found
            earliest_timestamp = None
            latest_timestamp = None
            data_coverage_warning = None

            with open(self.transactions_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Parse timestamp
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                    except:
                        continue

                    # Track actual data range (for all records, not just in range)
                    if earliest_timestamp is None or timestamp < earliest_timestamp:
                        earliest_timestamp = timestamp
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp

                    # Skip records outside requested time range but continue tracking range
                    if timestamp < cutoff_time or timestamp > end_time:
                        continue

                    total_requests += 1

                    # Success tracking
                    is_success = row['success'] == 'True'
                    if is_success:
                        successful_requests += 1

                    # Cost tracking
                    if row.get('cost_estimate'):
                        try:
                            cost = float(row['cost_estimate'])
                            total_cost += cost
                        except ValueError:
                            cost = 0.0
                    else:
                        cost = 0.0

                    # Time tracking
                    if row.get('total_time_ms'):
                        try:
                            time_ms = int(row['total_time_ms'])
                            total_time_ms += time_ms
                            time_count += 1
                            response_times.append(time_ms)
                        except ValueError:
                            pass

                    # Wait time compliance tracking
                    if row.get('max_wait_seconds') and row.get('total_time_ms'):
                        try:
                            max_wait_ms = int(row['max_wait_seconds']) * 1000
                            actual_time_ms = int(row['total_time_ms'])
                            wait_time_total += 1

                            if actual_time_ms <= max_wait_ms:
                                wait_time_met += 1
                            else:
                                wait_time_exceeded += 1
                        except ValueError:
                            pass

                    # Distributions
                    if row.get('provider_used'):
                        providers[row['provider_used']] += 1
                    if row.get('model_used'):
                        models[row['model_used']] += 1
                    if row.get('routing_method'):
                        routing_methods[row['routing_method']] += 1
                    if row.get('intelligence_level'):
                        intelligence_levels[row['intelligence_level']] += 1
                    if row.get('client_id'):
                        clients[row['client_id']] += 1

                    # Error tracking
                    if row.get('error_type'):
                        error_types[row['error_type']] += 1
                        error_details.append({
                            "timestamp": row['timestamp'],
                            "error_type": row['error_type'],
                            "error_message": row.get('error_message', ''),
                            "provider": row.get('requested_provider') or row.get('provider_used', ''),
                            "model": row.get('requested_model') or row.get('model_used', '')
                        })

                    # Time series (hourly buckets)
                    hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                    if is_success:
                        time_series[hour_key]["success"] += 1
                    else:
                        time_series[hour_key]["failed"] += 1
                    time_series[hour_key]["cost"] += cost

                    # API Key usage tracking
                    if row.get('api_key_used'):
                        api_key = row['api_key_used']
                        api_key_stats[api_key]["total_requests"] += 1

                        if is_success:
                            api_key_stats[api_key]["successful"] += 1
                        else:
                            api_key_stats[api_key]["failed"] += 1

                        api_key_stats[api_key]["total_cost"] += cost

                        # Track tokens
                        if row.get('input_tokens') and row.get('output_tokens'):
                            try:
                                tokens = int(row['input_tokens']) + int(row['output_tokens'])
                                api_key_stats[api_key]["total_tokens"] += tokens
                            except ValueError:
                                pass

                        # Track models used
                        if row.get('model_used'):
                            api_key_stats[api_key]["models_used"][row['model_used']] += 1

                        # Track response time
                        if row.get('total_time_ms'):
                            try:
                                api_key_stats[api_key]["response_time_sum"] += int(row['total_time_ms'])
                            except ValueError:
                                pass

                    # Wait time analysis
                    if row.get('max_wait_seconds'):
                        try:
                            max_wait = int(row['max_wait_seconds'])

                            # Determine bucket
                            if max_wait <= 10:
                                bucket = "0-10s"
                            elif max_wait <= 30:
                                bucket = "10-30s"
                            elif max_wait <= 60:
                                bucket = "30-60s"
                            elif max_wait <= 120:
                                bucket = "60-120s"
                            else:
                                bucket = "120s+"

                            wait_time_buckets[bucket]["count"] += 1
                            if is_success:
                                wait_time_buckets[bucket]["success"] += 1
                            else:
                                wait_time_buckets[bucket]["failed"] += 1
                        except ValueError:
                            pass

                    # Recent transactions (keep last 100)
                    if len(recent_transactions) < 100:
                        recent_transactions.append({
                            "timestamp": row['timestamp'],
                            "request_id": row['request_id'],
                            "client_id": row.get('client_id', ''),
                            "routing_method": row.get('routing_method', ''),
                            "intelligence_level": row.get('intelligence_level', ''),
                            "max_wait_seconds": row.get('max_wait_seconds', ''),
                            "success": is_success,
                            "provider": row.get('provider_used', ''),
                            "model": row.get('model_used', ''),
                            "api_key": row.get('api_key_used', ''),
                            "total_time_ms": row.get('total_time_ms', ''),
                            "cost": f"{cost:.6f}" if cost > 0 else '',
                            "error_type": row.get('error_type', ''),
                            "input_tokens": row.get('input_tokens', ''),
                            "output_tokens": row.get('output_tokens', '')
                        })

            # Sort time series
            sorted_time_series = []
            for hour in sorted(time_series.keys()):
                sorted_time_series.append({
                    "hour": hour,
                    "success": time_series[hour]["success"],
                    "failed": time_series[hour]["failed"],
                    "total": time_series[hour]["success"] + time_series[hour]["failed"],
                    "cost": round(time_series[hour]["cost"], 6)
                })

            # Reverse recent transactions (newest first)
            recent_transactions.reverse()

            # Calculate averages and percentiles
            avg_response_time = int(total_time_ms / time_count) if time_count > 0 else 0
            success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

            # Calculate P95 and P99
            p95_time = 0
            p99_time = 0
            if response_times:
                sorted_times = sorted(response_times)
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)
                p95_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
                p99_time = sorted_times[min(p99_index, len(sorted_times) - 1)]

            # Calculate wait time compliance rate
            wait_time_compliance_rate = (wait_time_met / wait_time_total * 100) if wait_time_total > 0 else 0

            # Determine data coverage status
            if total_requests == 0 and earliest_timestamp and latest_timestamp:
                # No data in requested range, but we have data elsewhere
                if cutoff_time < earliest_timestamp:
                    data_coverage_warning = f"No data available for the requested period. Available data starts from {earliest_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}."
                elif end_time > latest_timestamp:
                    data_coverage_warning = f"No data available for the requested period. Available data ends at {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}."
                else:
                    data_coverage_warning = f"No data available for the requested period. Available data: {earliest_timestamp.strftime('%Y-%m-%d')} to {latest_timestamp.strftime('%Y-%m-%d')}."
            elif total_requests > 0 and earliest_timestamp and latest_timestamp:
                # We have some data, but check if it's partial coverage
                if cutoff_time < earliest_timestamp or end_time > latest_timestamp:
                    actual_start = max(cutoff_time, earliest_timestamp).strftime('%Y-%m-%d %H:%M UTC')
                    actual_end = min(end_time, latest_timestamp).strftime('%Y-%m-%d %H:%M UTC')
                    data_coverage_warning = f"Partial data coverage. Showing data from {actual_start} to {actual_end}."

            # Process API key statistics
            api_keys_list = []
            for api_key, stats in api_key_stats.items():
                avg_time = int(stats["response_time_sum"] / stats["total_requests"]) if stats["total_requests"] > 0 else 0
                success_rate_key = (stats["successful"] / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0

                api_keys_list.append({
                    "api_key": api_key,
                    "total_requests": stats["total_requests"],
                    "successful": stats["successful"],
                    "failed": stats["failed"],
                    "success_rate": round(success_rate_key, 2),
                    "total_cost": round(stats["total_cost"], 6),
                    "total_tokens": stats["total_tokens"],
                    "avg_response_time_ms": avg_time,
                    "models_used": dict(stats["models_used"])
                })

            # Sort by total requests descending
            api_keys_list.sort(key=lambda x: x["total_requests"], reverse=True)

            # Process wait time analysis
            wait_time_analysis = []
            for bucket, data in wait_time_buckets.items():
                if data["count"] > 0:
                    bucket_success_rate = (data["success"] / data["count"] * 100)
                    wait_time_analysis.append({
                        "bucket": bucket,
                        "count": data["count"],
                        "success": data["success"],
                        "failed": data["failed"],
                        "success_rate": round(bucket_success_rate, 2)
                    })

            return {
                "overview": {
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": total_requests - successful_requests,
                    "success_rate": round(success_rate, 2),
                    "total_cost": round(total_cost, 6),
                    "avg_response_time_ms": avg_response_time,
                    "p95_response_time_ms": p95_time,
                    "p99_response_time_ms": p99_time,
                    "wait_time_compliance_rate": round(wait_time_compliance_rate, 2),
                    "wait_time_met": wait_time_met,
                    "wait_time_exceeded": wait_time_exceeded,
                    "time_window_hours": hours
                },
                "data_coverage": {
                    "warning": data_coverage_warning,
                    "earliest_available": earliest_timestamp.isoformat() if earliest_timestamp else None,
                    "latest_available": latest_timestamp.isoformat() if latest_timestamp else None,
                    "requested_start": cutoff_time.isoformat() if cutoff_time else None,
                    "requested_end": end_time.isoformat() if end_time else None,
                    "timezone_info": "All logs are stored in UTC. Timestamps can be displayed in different timezones using the timezone selector."
                },
                "distributions": {
                    "providers": dict(providers),
                    "models": dict(models),
                    "routing_methods": dict(routing_methods),
                    "intelligence_levels": dict(intelligence_levels),
                    "clients": dict(clients)
                },
                "time_series": sorted_time_series,
                "recent_transactions": recent_transactions,
                "errors": {
                    "types": dict(error_types),
                    "details": error_details[:50]  # Last 50 errors
                },
                "api_keys": api_keys_list,
                "wait_time_analysis": wait_time_analysis
            }

        except Exception as e:
            logger.error("Failed to process transaction statistics", error=str(e))
            return self._empty_transaction_data()

    async def _get_daily_statistics(self) -> Dict[str, Any]:
        """Get daily aggregated statistics"""
        if not self.daily_stats_file.exists():
            return {}

        try:
            with open(self.daily_stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Sort by date and return last 30 days
            sorted_dates = sorted(data.keys(), reverse=True)[:30]
            return {date: data[date] for date in sorted_dates}

        except Exception as e:
            logger.error("Failed to load daily statistics", error=str(e))
            return {}

    def _empty_transaction_data(self) -> Dict[str, Any]:
        """Return empty data structure when no transactions exist"""
        return {
            "overview": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "total_cost": 0.0,
                "avg_response_time_ms": 0,
                "time_window_hours": 24
            },
            "distributions": {
                "providers": {},
                "models": {},
                "routing_methods": {},
                "intelligence_levels": {},
                "clients": {}
            },
            "time_series": [],
            "recent_transactions": [],
            "errors": {
                "types": {},
                "details": []
            },
            "api_keys": [],
            "wait_time_analysis": []
        }

    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health information"""
        return {
            "transaction_log_exists": self.transactions_file.exists(),
            "transaction_log_size": self.transactions_file.stat().st_size if self.transactions_file.exists() else 0,
            "daily_stats_exists": self.daily_stats_file.exists(),
            "log_directory": str(self.log_dir),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global instance
_dashboard_processor: Optional[DashboardDataProcessor] = None


def init_dashboard_processor(log_dir: str = "logs") -> DashboardDataProcessor:
    """Initialize the global dashboard processor"""
    global _dashboard_processor
    _dashboard_processor = DashboardDataProcessor(log_dir=log_dir)
    return _dashboard_processor


def get_dashboard_processor() -> Optional[DashboardDataProcessor]:
    """Get the global dashboard processor instance"""
    return _dashboard_processor
