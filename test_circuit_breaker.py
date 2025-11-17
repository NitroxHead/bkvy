#!/usr/bin/env python3
"""
Test script for circuit breaker functionality
"""

import asyncio
import aiohttp
import json
from datetime import datetime

BASE_URL = "http://localhost:10006"


async def test_circuit_status():
    """Test getting circuit breaker status"""
    print("\n" + "="*80)
    print("TEST: Circuit Breaker Status")
    print("="*80)

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/circuits/status") as response:
            data = await response.json()

            print(f"\n✅ Circuit Breaker Enabled: {data.get('enabled')}")
            print(f"   Total Circuits: {data.get('total_circuits', 0)}")
            print(f"   Open Circuits: {data.get('open_circuits', 0)}")
            print(f"   Half-Open Circuits: {data.get('half_open_circuits', 0)}")
            print(f"   Closed Circuits: {data.get('closed_circuits', 0)}")

            return data


async def test_provider_health():
    """Test provider health endpoint"""
    print("\n" + "="*80)
    print("TEST: Provider Health")
    print("="*80)

    providers = ["gemini", "openai", "anthropic", "ollama"]

    async with aiohttp.ClientSession() as session:
        for provider in providers:
            try:
                async with session.get(f"{BASE_URL}/circuits/provider/{provider}") as response:
                    if response.status == 200:
                        data = await response.json()

                        print(f"\n📊 {provider.upper()}")
                        print(f"   Overall Health: {data.get('overall_health', 'unknown')}")
                        print(f"   Total Circuits: {data.get('total_circuits', 0)}")
                        print(f"   Open: {data.get('open_circuits', 0)} | "
                              f"Half-Open: {data.get('half_open_circuits', 0)} | "
                              f"Closed: {data.get('closed_circuits', 0)}")

                        if data.get('failure_pattern'):
                            print(f"   ⚠️  Failure Pattern: {data['failure_pattern']}")
                        if data.get('recommended_action'):
                            print(f"   💡 Recommended: {data['recommended_action']}")
            except Exception as e:
                print(f"\n❌ {provider.upper()}: {str(e)}")


async def test_llm_request():
    """Test making an LLM request"""
    print("\n" + "="*80)
    print("TEST: LLM Request with Circuit Breaker")
    print("="*80)

    request_data = {
        "client_id": "test_circuit_breaker",
        "intelligence_level": "low",
        "max_wait_seconds": 30,
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "debug": True  # Enable debug mode to see routing details
    }

    async with aiohttp.ClientSession() as session:
        print("\n🚀 Sending request...")
        print(f"   Intelligence Level: {request_data['intelligence_level']}")
        print(f"   Max Wait: {request_data['max_wait_seconds']}s")

        start = datetime.now()

        try:
            async with session.post(
                f"{BASE_URL}/llm/intelligence",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                elapsed = (datetime.now() - start).total_seconds()
                data = await response.json()

                if data.get("success"):
                    print(f"\n✅ Request Successful ({elapsed:.2f}s)")
                    print(f"   Provider: {data.get('provider_used')}")
                    print(f"   Model: {data.get('model_used')}")
                    print(f"   API Key: {data.get('api_key_used')}")
                    print(f"   Decision Reason: {data.get('decision_reason')}")

                    if data.get('metadata'):
                        metadata = data['metadata']
                        print(f"\n📊 Performance:")
                        print(f"   Rate Limit Wait: {metadata.get('rate_limit_wait_ms', 0)}ms")
                        print(f"   Queue Wait: {metadata.get('queue_wait_ms', 0)}ms")
                        print(f"   API Response: {metadata.get('api_response_time_ms', 0)}ms")
                        print(f"   Total Time: {metadata.get('total_completion_time_ms', 0)}ms")
                else:
                    print(f"\n❌ Request Failed ({elapsed:.2f}s)")
                    print(f"   Error: {data.get('message')}")
                    print(f"   Error Code: {data.get('error_code')}")

                return data

        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            print(f"\n💥 Request Exception ({elapsed:.2f}s)")
            print(f"   Error: {str(e)}")
            raise


async def test_circuits_summary():
    """Test circuits summary endpoint"""
    print("\n" + "="*80)
    print("TEST: Circuits Summary")
    print("="*80)

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/circuits/summary") as response:
            data = await response.json()

            if data.get('enabled'):
                print("\n✅ Circuit Breaker Summary:")

                for provider, health in data.get('providers', {}).items():
                    status_emoji = {
                        'healthy': '✅',
                        'degraded': '⚠️',
                        'unhealthy': '❌'
                    }.get(health.get('overall_health'), '❓')

                    print(f"\n{status_emoji} {provider.upper()}")
                    print(f"   Health: {health.get('overall_health', 'unknown')}")
                    print(f"   Circuits: {health.get('closed_circuits', 0)} closed, "
                          f"{health.get('open_circuits', 0)} open, "
                          f"{health.get('half_open_circuits', 0)} half-open")

                    if health.get('failure_pattern'):
                        print(f"   Pattern: {health['failure_pattern']}")


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CIRCUIT BREAKER TEST SUITE")
    print("="*80)
    print(f"Testing against: {BASE_URL}")
    print(f"Started at: {datetime.now().isoformat()}")

    try:
        # Test 1: Circuit Status
        await test_circuit_status()

        # Test 2: Provider Health
        await test_provider_health()

        # Test 3: Circuits Summary
        await test_circuits_summary()

        # Test 4: LLM Request
        await test_llm_request()

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED")
        print("="*80 + "\n")

    except aiohttp.ClientConnectorError:
        print("\n" + "="*80)
        print("❌ ERROR: Cannot connect to server")
        print("="*80)
        print("\nMake sure the server is running:")
        print("  export CIRCUIT_BREAKER_ENABLED=true")
        print("  python3 main.py")
        print()

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR: {str(e)}")
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())
