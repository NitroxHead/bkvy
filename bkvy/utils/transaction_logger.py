"""
Transaction logging for request statistics
"""

import csv
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .logging import setup_logging

logger = setup_logging()


@dataclass
class TransactionRecord:
    """Data structure for a single transaction record"""
    timestamp: str
    request_id: str
    client_id: str
    routing_method: str  # intelligence, scenario, direct
    intelligence_level: Optional[str] = None  # low, medium, high
    scenario: Optional[str] = None
    requested_provider: Optional[str] = None
    requested_model: Optional[str] = None
    max_wait_seconds: Optional[int] = None  # Requested maximum wait time

    # What actually happened
    success: bool = False
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    api_key_used: Optional[str] = None

    # Performance metrics
    total_time_ms: Optional[int] = None
    queue_wait_ms: Optional[int] = None
    rate_limit_wait_ms: Optional[int] = None

    # Usage metrics
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_estimate: Optional[float] = None

    # Results and reasons
    finish_reason: Optional[str] = None
    decision_reason: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Fallback info
    fallback_attempts: int = 0
    alternatives_tried: int = 0


class TransactionLogger:
    """Handles CSV logging of detailed transaction statistics"""

    def __init__(self, enabled: bool = True, log_dir: str = "logs"):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.csv_file = self.log_dir / "transactions.csv"
        self._write_lock = asyncio.Lock()
        self._headers_written = False

        if self.enabled:
            self.log_dir.mkdir(exist_ok=True)
            self._ensure_headers()

    def _ensure_headers(self):
        """Ensure CSV headers are written"""
        if not self.csv_file.exists():
            # Create file with headers
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
                writer.writeheader()
            self._headers_written = True

    def _get_fieldnames(self) -> list:
        """Get CSV fieldnames from TransactionRecord"""
        return list(TransactionRecord.__dataclass_fields__.keys())

    async def log_transaction(self, record: TransactionRecord):
        """Log a transaction record to CSV"""
        if not self.enabled:
            return

        try:
            async with self._write_lock:
                # Convert record to dict and handle None values
                record_dict = asdict(record)

                # Write to CSV
                with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
                    writer.writerow(record_dict)

                logger.debug("Transaction logged", request_id=record.request_id)

        except Exception as e:
            logger.error("Failed to log transaction",
                        request_id=record.request_id,
                        error=str(e))

    def create_record(self, request_id: str, client_id: str, routing_method: str) -> TransactionRecord:
        """Create a new transaction record with defaults"""
        return TransactionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request_id,
            client_id=client_id,
            routing_method=routing_method
        )

    async def get_stats_summary(self) -> Dict[str, Any]:
        """Get basic statistics summary from the CSV file"""
        if not self.enabled or not self.csv_file.exists():
            return {"enabled": False}

        try:
            total_requests = 0
            successful_requests = 0
            routing_methods = {}
            providers_used = {}
            intelligence_levels = {}
            errors = {}
            total_cost = 0.0

            with open(self.csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    total_requests += 1

                    if row['success'] == 'True':
                        successful_requests += 1

                    # Count routing methods
                    method = row['routing_method']
                    routing_methods[method] = routing_methods.get(method, 0) + 1

                    # Count providers used
                    if row['provider_used']:
                        provider = row['provider_used']
                        providers_used[provider] = providers_used.get(provider, 0) + 1

                    # Count intelligence levels
                    if row['intelligence_level']:
                        level = row['intelligence_level']
                        intelligence_levels[level] = intelligence_levels.get(level, 0) + 1

                    # Count errors
                    if row['error_type']:
                        error = row['error_type']
                        errors[error] = errors.get(error, 0) + 1

                    # Sum costs
                    if row['cost_estimate']:
                        try:
                            total_cost += float(row['cost_estimate'])
                        except ValueError:
                            pass

            return {
                "enabled": True,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "routing_methods": routing_methods,
                "providers_used": providers_used,
                "intelligence_levels": intelligence_levels,
                "errors": errors,
                "total_cost_estimate": round(total_cost, 6),
                "log_file": str(self.csv_file)
            }

        except Exception as e:
            logger.error("Failed to generate stats summary", error=str(e))
            return {"enabled": True, "error": str(e)}


# Global instance that can be configured
_transaction_logger: Optional[TransactionLogger] = None


def init_transaction_logger(enabled: bool = True, log_dir: str = "logs") -> TransactionLogger:
    """Initialize the global transaction logger"""
    global _transaction_logger
    _transaction_logger = TransactionLogger(enabled=enabled, log_dir=log_dir)
    return _transaction_logger


def get_transaction_logger() -> Optional[TransactionLogger]:
    """Get the global transaction logger instance"""
    return _transaction_logger


def is_transaction_logging_enabled() -> bool:
    """Check if transaction logging is enabled"""
    return _transaction_logger is not None and _transaction_logger.enabled