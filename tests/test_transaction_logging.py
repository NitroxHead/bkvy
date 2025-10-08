#!/usr/bin/env python3
"""
Simple test for transaction logging functionality
"""

import asyncio
import tempfile
import csv
from pathlib import Path

from bkvy.utils.transaction_logger import TransactionLogger, TransactionRecord


async def test_transaction_logging():
    """Test transaction logging functionality"""
    print("Testing transaction logging...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize transaction logger
        logger = TransactionLogger(enabled=True, log_dir=temp_dir)

        # Create test transaction records
        record1 = logger.create_record(
            request_id="test-001",
            client_id="test-client",
            routing_method="intelligence"
        )
        record1.intelligence_level = "high"
        record1.success = True
        record1.provider_used = "openai"
        record1.model_used = "gpt-4o"
        record1.api_key_used = "test_key_1"
        record1.total_time_ms = 1500
        record1.input_tokens = 100
        record1.output_tokens = 50
        record1.cost_estimate = 0.00375
        record1.decision_reason = "cheapest_within_estimate"
        record1.finish_reason = "stop"

        record2 = logger.create_record(
            request_id="test-002",
            client_id="test-client",
            routing_method="scenario"
        )
        record2.scenario = "cost_optimized"
        record2.success = False
        record2.error_type = "rate_limited"
        record2.error_message = "Rate limit exceeded"
        record2.total_time_ms = 500
        record2.fallback_attempts = 2
        record2.alternatives_tried = 3

        # Log the records
        await logger.log_transaction(record1)
        await logger.log_transaction(record2)

        print(f"✓ Logged 2 transactions to {logger.csv_file}")

        # Verify CSV file was created and has correct content
        assert logger.csv_file.exists(), "CSV file was not created"

        # Read and verify content
        with open(logger.csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

        # Verify first record
        row1 = rows[0]
        assert row1['request_id'] == 'test-001'
        assert row1['routing_method'] == 'intelligence'
        assert row1['intelligence_level'] == 'high'
        assert row1['success'] == 'True'
        assert row1['provider_used'] == 'openai'
        assert row1['model_used'] == 'gpt-4o'
        assert row1['input_tokens'] == '100'
        assert row1['output_tokens'] == '50'
        assert row1['cost_estimate'] == '0.00375'

        print("✓ First record verified")

        # Verify second record
        row2 = rows[1]
        assert row2['request_id'] == 'test-002'
        assert row2['routing_method'] == 'scenario'
        assert row2['scenario'] == 'cost_optimized'
        assert row2['success'] == 'False'
        assert row2['error_type'] == 'rate_limited'
        assert row2['fallback_attempts'] == '2'
        assert row2['alternatives_tried'] == '3'

        print("✓ Second record verified")

        # Test statistics summary
        stats = await logger.get_stats_summary()
        assert stats['enabled'] is True
        assert stats['total_requests'] == 2
        assert stats['successful_requests'] == 1
        assert stats['success_rate'] == 0.5
        assert 'intelligence' in stats['routing_methods']
        assert 'scenario' in stats['routing_methods']
        assert 'openai' in stats['providers_used']
        assert 'rate_limited' in stats['errors']

        print("✓ Statistics summary verified")
        print(f"✓ Stats: {stats}")

        print("\n🎉 All tests passed!")

        return True


async def test_disabled_logging():
    """Test that disabled logging works correctly"""
    print("Testing disabled transaction logging...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize disabled transaction logger
        logger = TransactionLogger(enabled=False, log_dir=temp_dir)

        # Create test transaction record
        record = logger.create_record(
            request_id="test-disabled",
            client_id="test-client",
            routing_method="intelligence"
        )

        # Try to log the record (should be ignored)
        await logger.log_transaction(record)

        # Verify no CSV file was created
        assert not logger.csv_file.exists(), "CSV file should not be created when disabled"

        # Test statistics (should return disabled status)
        stats = await logger.get_stats_summary()
        assert stats['enabled'] is False

        print("✓ Disabled logging test passed")

        return True


async def main():
    """Run all tests"""
    print("=" * 50)
    print("Transaction Logging Test Suite")
    print("=" * 50)

    try:
        await test_transaction_logging()
        print()
        await test_disabled_logging()

        print()
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nTransaction logging feature is ready to use!")
        print("\nUsage:")
        print("1. Set environment variable TRANSACTION_LOGGING=true (default)")
        print("2. Optionally set TRANSACTION_LOG_DIR=path/to/logs (default: logs)")
        print("3. Start the server: python main.py")
        print("4. View statistics: GET /statistics/summary")
        print("5. Check CSV file: logs/transactions.csv")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)