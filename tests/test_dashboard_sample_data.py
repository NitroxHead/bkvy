#!/usr/bin/env python3
"""
Generate sample transaction data for testing the dashboard
"""

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Sample data configuration
PROVIDERS = ["openai", "anthropic", "gemini", "ollama"]
MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "ollama": ["llama3.1:8b", "qwen2.5:14b"]
}
ROUTING_METHODS = ["intelligence", "scenario", "direct"]
INTELLIGENCE_LEVELS = ["low", "medium", "high"]
CLIENTS = ["web-app", "mobile-app", "api-client", "batch-processor"]
ERROR_TYPES = ["rate_limited", "model_not_found", "timeout", "invalid_request", "no_models_available"]

def generate_sample_transactions(count=500, hours_back=48):
    """Generate sample transaction records"""
    transactions = []
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours_back)

    for i in range(count):
        # Generate random timestamp within range
        time_delta = random.random() * (end_time - start_time).total_seconds()
        timestamp = start_time + timedelta(seconds=time_delta)

        # Random provider and model
        provider = random.choice(PROVIDERS)
        model = random.choice(MODELS[provider])

        # Random routing method
        routing_method = random.choice(ROUTING_METHODS)

        # Success rate: 85%
        success = random.random() < 0.85

        # Generate transaction
        transaction = {
            "timestamp": timestamp.isoformat(),
            "request_id": f"req_{i:06d}",
            "client_id": random.choice(CLIENTS),
            "routing_method": routing_method,
            "intelligence_level": random.choice(INTELLIGENCE_LEVELS) if routing_method == "intelligence" else "",
            "scenario": f"scenario_{random.randint(1,3)}" if routing_method == "scenario" else "",
            "requested_provider": provider if routing_method == "direct" else "",
            "requested_model": model if routing_method == "direct" else "",
            "max_wait_seconds": random.randint(10, 120),
            "success": success,
            "provider_used": provider if success else "",
            "model_used": model if success else "",
            "api_key_used": f"{provider}_key_{random.randint(1,3)}" if success else "",
            "total_time_ms": random.randint(500, 5000) if success else random.randint(100, 2000),
            "queue_wait_ms": random.randint(0, 1000) if success else "",
            "rate_limit_wait_ms": random.randint(0, 500) if success else "",
            "input_tokens": random.randint(50, 2000) if success else "",
            "output_tokens": random.randint(100, 3000) if success else "",
            "cost_estimate": round(random.uniform(0.0001, 0.05), 6) if success else "",
            "finish_reason": "stop" if success else "",
            "decision_reason": f"cheapest_within_estimate_after_{random.randint(1,3)}_attempts" if success else "",
            "error_type": random.choice(ERROR_TYPES) if not success else "",
            "error_message": f"Sample error message: {random.choice(ERROR_TYPES)}" if not success else "",
            "fallback_attempts": random.randint(0, 2) if not success else 0,
            "alternatives_tried": random.randint(1, 5)
        }

        transactions.append(transaction)

    # Sort by timestamp
    transactions.sort(key=lambda x: x["timestamp"])

    return transactions


def write_transactions_csv(transactions, output_file="logs/transactions.csv"):
    """Write transactions to CSV file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    fieldnames = list(transactions[0].keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)

    print(f"✅ Generated {len(transactions)} sample transactions")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 File size: {output_path.stat().st_size / 1024:.2f} KB")


def main():
    """Generate sample data for testing"""
    print("🔧 Generating sample transaction data for dashboard testing...")

    # Generate 500 transactions over the last 48 hours
    transactions = generate_sample_transactions(count=500, hours_back=48)

    # Write to CSV
    write_transactions_csv(transactions)

    # Print summary statistics
    successful = sum(1 for t in transactions if t["success"])
    success_rate = (successful / len(transactions)) * 100

    print(f"\n📈 Summary:")
    print(f"   Total requests: {len(transactions)}")
    print(f"   Successful: {successful} ({success_rate:.1f}%)")
    print(f"   Failed: {len(transactions) - successful} ({100 - success_rate:.1f}%)")
    print(f"   Time range: {transactions[0]['timestamp']} to {transactions[-1]['timestamp']}")
    print(f"\n🚀 You can now test the dashboard at: http://localhost:10006/dashboard")
    print(f"   (Make sure to set DASHBOARD_ENABLED=true and TRANSACTION_LOGGING=true)")


if __name__ == "__main__":
    main()
