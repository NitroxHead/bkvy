#!/usr/bin/env python3
"""
Advanced circuit breaker test - Tests failure scenarios
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import time


BASE_URL = "http://localhost:10006"


async def test_consecutive_failures():
    """Test circuit opening after 3 consecutive failures"""
    print("\n" + "="*80)
    print("TEST: Consecutive Failures (Circuit Opening)")
    print("="*80)

    # Use an invalid API key to force failures
    request_data = {
        "provider": "openai",
        "model": "gpt-4o-mini-2024-07-18",
        "api_key_override": "sk-invalid-key-to-trigger-failure",  # Invalid key
        "messages": [{"role": "user", "content": "test"}],
        "debug": True
    }

    async with aiohttp.ClientSession() as session:
        print("\n🔥 Sending 3 requests with invalid API key to trigger circuit opening...")

        for i in range(3):
            print(f"\n  Attempt {i+1}/3...")
            try:
                async with session.post(
                    f"{BASE_URL}/llm/direct",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    data = await response.json()
                    print(f"    Status: {response.status}")
                    print(f"    Success: {data.get('success')}")
                    if not data.get('success'):
                        print(f"    Error: {data.get('message', 'Unknown')[:100]}")
            except Exception as e:
                print(f"    Exception: {str(e)[:100]}")

            await asyncio.sleep(1)  # Small delay between requests

        # Check circuit state
        print("\n📊 Checking circuit state after failures...")
        await asyncio.sleep(1)

        async with session.get(f"{BASE_URL}/circuits/status") as response:
            data = await response.json()

            # Find OpenAI circuits
            openai_circuits = {k: v for k, v in data.get('circuits', {}).items()
                              if v['provider'] == 'openai'}

            for key, circuit in openai_circuits.items():
                print(f"\n  Circuit: {key}")
                print(f"    State: {circuit['state']}")
                print(f"    Consecutive Failures: {circuit['consecutive_failures']}")
                print(f"    Failure Count: {circuit['failure_count']}")
                print(f"    Failure History Length: {len(circuit.get('failure_history', []))}")
                if circuit['state'] == 'open':
                    print(f"    ✅ Circuit OPENED as expected!")
                    print(f"    Next Test Time: {circuit.get('next_test_time')}")
                    print(f"    Backoff Level: {circuit.get('backoff_level')}")


async def test_sliding_window():
    """Test sliding window threshold (5 failures in 10 minutes)"""
    print("\n" + "="*80)
    print("TEST: Sliding Window Threshold")
    print("="*80)
    print("\nNote: Using shorter window (60s) for faster testing")
    print("In production, this would be 600s (10 minutes)")

    # We'll simulate intermittent failures
    # For testing, we'll space them out to avoid consecutive threshold

    print("\n🔥 Simulating intermittent failures over time...")
    print("    (Pattern: Fail → Success → Fail → Success → Fail...)")

    async with aiohttp.ClientSession() as session:
        failure_count = 0

        for i in range(8):
            if i % 2 == 0:  # Even numbers = failures
                # Invalid request
                request_data = {
                    "provider": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "api_key_override": "sk-invalid",
                    "messages": [{"role": "user", "content": "test"}]
                }
                expected = "FAIL"
            else:  # Odd numbers = successes
                # Valid request
                request_data = {
                    "client_id": "test_sliding",
                    "intelligence_level": "low",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "options": {"max_tokens": 5}
                }
                expected = "SUCCESS"

            print(f"\n  Request {i+1}/8 (expecting {expected})...")

            try:
                endpoint = "/llm/direct" if i % 2 == 0 else "/llm/intelligence"
                async with session.post(
                    f"{BASE_URL}{endpoint}",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    data = await response.json()
                    success = data.get('success')

                    if not success:
                        failure_count += 1
                        print(f"    ❌ Failed (total failures: {failure_count})")
                    else:
                        print(f"    ✅ Success")
            except Exception as e:
                print(f"    Exception: {str(e)[:100]}")

            await asyncio.sleep(2)  # Small delay

        # Check anthropic circuits
        print("\n📊 Checking Anthropic circuit states...")
        async with session.get(f"{BASE_URL}/circuits/provider/anthropic") as response:
            data = await response.json()
            print(f"\n  Overall Health: {data.get('overall_health')}")
            print(f"  Total Circuits: {data.get('total_circuits')}")
            print(f"  Open Circuits: {data.get('open_circuits')}")
            print(f"  Failure Pattern: {data.get('failure_pattern')}")


async def test_half_open_transition():
    """Test OPEN → HALF_OPEN transition"""
    print("\n" + "="*80)
    print("TEST: Circuit Recovery (HALF_OPEN)")
    print("="*80)

    # First, check if any circuits are OPEN
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/circuits/status") as response:
            data = await response.json()

            open_circuits = [k for k, v in data.get('circuits', {}).items()
                           if v['state'] == 'open']

            if open_circuits:
                print(f"\n✅ Found {len(open_circuits)} OPEN circuit(s)")

                for circuit_key in open_circuits[:1]:  # Test first one
                    circuit = data['circuits'][circuit_key]
                    print(f"\n  Circuit: {circuit_key}")
                    print(f"  Current State: {circuit['state']}")
                    print(f"  Next Test Time: {circuit.get('next_test_time')}")

                    # Calculate wait time
                    if circuit.get('next_test_time'):
                        next_test = datetime.fromisoformat(circuit['next_test_time'].replace('Z', '+00:00'))
                        now = datetime.now(next_test.tzinfo)
                        wait_seconds = max(0, (next_test - now).total_seconds())

                        print(f"  Wait Time: {wait_seconds:.1f} seconds")

                        if wait_seconds > 60:
                            print(f"  ⚠️  Wait time too long for test ({wait_seconds:.0f}s)")
                            print(f"  💡 Use manual reset: curl -X POST {BASE_URL}/circuits/reset/{circuit['provider']}/{circuit['model']}/{circuit['api_key_id']}")
                        elif wait_seconds > 0:
                            print(f"  ⏳ Waiting {wait_seconds:.1f}s for circuit to be ready...")
                            await asyncio.sleep(wait_seconds + 1)

                            # Check state again
                            async with session.get(f"{BASE_URL}/circuits/status") as resp:
                                new_data = await resp.json()
                                new_state = new_data['circuits'][circuit_key]['state']
                                print(f"  New State: {new_state}")

                                if new_state == 'half_open':
                                    print(f"  ✅ Circuit transitioned to HALF_OPEN!")
            else:
                print("\n  No OPEN circuits found (all circuits are healthy)")
                print("  💡 Run test_consecutive_failures() first to open a circuit")


async def test_provider_health_aggregation():
    """Test provider-level health aggregation"""
    print("\n" + "="*80)
    print("TEST: Provider Health Aggregation")
    print("="*80)

    async with aiohttp.ClientSession() as session:
        for provider in ["gemini", "openai", "anthropic", "ollama"]:
            async with session.get(f"{BASE_URL}/circuits/provider/{provider}") as response:
                data = await response.json()

                print(f"\n📊 {provider.upper()}")
                print(f"  Overall Health: {data.get('overall_health')}")
                print(f"  Total Circuits: {data.get('total_circuits')}")
                print(f"  Open: {data.get('open_circuits')} | Half-Open: {data.get('half_open_circuits')} | Closed: {data.get('closed_circuits')}")

                if data.get('failure_pattern'):
                    print(f"  ⚠️  Failure Pattern: {data.get('failure_pattern')}")
                if data.get('recommended_action'):
                    print(f"  💡 Recommended: {data.get('recommended_action')}")
                if data.get('should_skip_provider'):
                    print(f"  🚫 Should Skip Provider: True")


async def test_circuit_reset():
    """Test manual circuit reset"""
    print("\n" + "="*80)
    print("TEST: Manual Circuit Reset")
    print("="*80)

    async with aiohttp.ClientSession() as session:
        # Find any non-closed circuit
        async with session.get(f"{BASE_URL}/circuits/status") as response:
            data = await response.json()

            non_closed = [k for k, v in data.get('circuits', {}).items()
                         if v['state'] != 'closed']

            if non_closed:
                circuit_key = non_closed[0]
                circuit = data['circuits'][circuit_key]

                print(f"\n🔧 Resetting circuit: {circuit_key}")
                print(f"  Current State: {circuit['state']}")
                print(f"  Failure Count: {circuit['failure_count']}")

                # Reset it
                reset_url = f"{BASE_URL}/circuits/reset/{circuit['provider']}/{circuit['model']}/{circuit['api_key_id']}"
                async with session.post(reset_url) as resp:
                    result = await resp.json()
                    print(f"\n  Reset Result: {result.get('success')}")
                    print(f"  Message: {result.get('message')}")

                # Verify reset
                await asyncio.sleep(1)
                async with session.get(f"{BASE_URL}/circuits/status") as resp:
                    new_data = await resp.json()
                    new_circuit = new_data['circuits'][circuit_key]

                    print(f"\n  After Reset:")
                    print(f"    State: {new_circuit['state']}")
                    print(f"    Failure Count: {new_circuit['failure_count']}")
                    print(f"    Consecutive Failures: {new_circuit['consecutive_failures']}")

                    if new_circuit['state'] == 'closed' and new_circuit['failure_count'] == 0:
                        print(f"    ✅ Circuit successfully reset!")
            else:
                print("\n  All circuits are CLOSED (nothing to reset)")


async def test_failure_history_persistence():
    """Test that failure history is properly saved"""
    print("\n" + "="*80)
    print("TEST: Failure History Persistence")
    print("="*80)

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/circuits/status") as response:
            data = await response.json()

            circuits_with_failures = {k: v for k, v in data.get('circuits', {}).items()
                                     if v['failure_count'] > 0}

            if circuits_with_failures:
                print(f"\n✅ Found {len(circuits_with_failures)} circuit(s) with failures")

                for key, circuit in list(circuits_with_failures.items())[:3]:
                    print(f"\n  Circuit: {key}")
                    print(f"  Failure Count: {circuit['failure_count']}")
                    print(f"  Failure History Length: {len(circuit.get('failure_history', []))}")

                    # Show first few failures
                    history = circuit.get('failure_history', [])
                    if history:
                        print(f"\n  Recent Failures:")
                        for i, failure in enumerate(history[:3], 1):
                            print(f"    {i}. Type: {failure.get('failure_type')}")
                            print(f"       Time: {failure.get('timestamp')}")
                            print(f"       Error: {failure.get('error_message', '')[:60]}...")

                        # Verify it's in the persisted file
                        print(f"\n  💾 Checking persistence...")
                        import os
                        circuit_file = f"circuit_states/{key}.json"
                        if os.path.exists(circuit_file):
                            print(f"    ✅ Circuit state file exists: {circuit_file}")
                            with open(circuit_file, 'r') as f:
                                saved_data = json.load(f)
                                saved_history = saved_data.get('failure_history', [])
                                print(f"    ✅ Saved failure history length: {len(saved_history)}")
                        else:
                            print(f"    ❌ Circuit state file not found!")
            else:
                print("\n  No circuits with failures found")
                print("  💡 This test requires failures to have occurred")


async def main():
    """Run all failure tests"""
    print("\n" + "="*80)
    print("CIRCUIT BREAKER FAILURE TESTS")
    print("="*80)
    print(f"Testing against: {BASE_URL}")
    print(f"Started at: {datetime.now().isoformat()}")

    try:
        # Check server
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    print("❌ Server health check failed")
                    return

        # Run tests
        await test_consecutive_failures()
        await asyncio.sleep(2)

        await test_sliding_window()
        await asyncio.sleep(2)

        await test_provider_health_aggregation()
        await asyncio.sleep(2)

        await test_failure_history_persistence()
        await asyncio.sleep(2)

        await test_half_open_transition()
        await asyncio.sleep(2)

        await test_circuit_reset()

        print("\n" + "="*80)
        print("✅ ALL FAILURE TESTS COMPLETED")
        print("="*80 + "\n")

    except aiohttp.ClientConnectorError:
        print("\n❌ Cannot connect to server")
        print("Make sure server is running: python3 main.py\n")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
