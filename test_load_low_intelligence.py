#!/usr/bin/env python3
"""
Load test for low intelligence requests

Sends 50 concurrent low intelligence requests to the bkvy
and collects performance statistics.
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev


@dataclass
class RequestResult:
    """Result of a single request"""
    request_id: int
    success: bool
    response_time_ms: float
    model_used: str = None
    provider_used: str = None
    error: str = None
    status_code: int = None
    content_length: int = 0
    tokens_used: int = 0


@dataclass
class TestStatistics:
    """Overall test statistics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    total_test_time_ms: float
    
    # Response time statistics (milliseconds)
    avg_response_time_ms: float
    median_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    std_dev_response_time_ms: float
    
    # Throughput
    requests_per_second: float
    
    # Model/Provider usage
    models_used: Dict[str, int]
    providers_used: Dict[str, int]
    
    # Error analysis
    error_types: Dict[str, int]
    status_codes: Dict[int, int]
    
    # Token usage
    total_tokens: int
    avg_tokens_per_request: float


async def make_request(session: aiohttp.ClientSession, request_id: int, 
                      base_url: str = "http://localhost:10006") -> RequestResult:
    """Make a single low intelligence request"""
    
    # Using the same message for all requests to match your curl example
    
    payload = {
        "client_id": "bulletproof_test",
        "intelligence_level": "low",
        "max_wait_seconds": 8,
        "messages": [{"role": "user", "content": "say favoire word"}],
        "options": {
            "max_tokens": 10,
            "temperature": 0.1
        },
        "debug": True
    }
    
    start_time = time.time()
    
    try:
        async with session.post(
            f"{base_url}/llm/intelligence",
            json=payload
        ) as response:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            response_text = await response.text()
            content_length = len(response_text)
            
            if response.status == 200:
                data = json.loads(response_text)
                
                # Extract information from response
                model_used = data.get("model_used", "unknown")
                provider_used = data.get("provider_used", "unknown")
                
                # Count tokens if available
                tokens_used = 0
                if "response" in data and "usage" in data["response"]:
                    usage = data["response"]["usage"]
                    tokens_used = usage.get("total_tokens", 0)
                elif "usage" in data:
                    usage = data["usage"]
                    tokens_used = usage.get("total_tokens", 0)
                
                return RequestResult(
                    request_id=request_id,
                    success=True,
                    response_time_ms=response_time_ms,
                    model_used=model_used,
                    provider_used=provider_used,
                    status_code=response.status,
                    content_length=content_length,
                    tokens_used=tokens_used
                )
            else:
                return RequestResult(
                    request_id=request_id,
                    success=False,
                    response_time_ms=response_time_ms,
                    error=f"HTTP {response.status}: {response_text[:200]}",
                    status_code=response.status,
                    content_length=content_length
                )
                
    except Exception as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        return RequestResult(
            request_id=request_id,
            success=False,
            response_time_ms=response_time_ms,
            error=str(e),
            status_code=0
        )


async def run_load_test(num_requests: int = 50, base_url: str = "http://localhost:10006") -> TestStatistics:
    """Run the load test with specified number of concurrent requests"""
    
    print(f"Starting load test with {num_requests} concurrent low intelligence requests...")
    print(f"Target URL: {base_url}")
    print("-" * 60)
    
    start_time = time.time()
    
    # Create aiohttp session with connection limits and no timeout
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=100)
    timeout = aiohttp.ClientTimeout(total=None)  # No timeout - wait indefinitely
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        # Create all request tasks
        tasks = [
            make_request(session, i, base_url) 
            for i in range(num_requests)
        ]
        
        # Execute all requests concurrently
        print(f"Sending {num_requests} requests concurrently...")
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_test_time_ms = (end_time - start_time) * 1000
    
    # Calculate statistics
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    response_times = [r.response_time_ms for r in results]
    successful_response_times = [r.response_time_ms for r in successful_results]
    
    # Model/Provider usage counting
    models_used = {}
    providers_used = {}
    for result in successful_results:
        if result.model_used:
            models_used[result.model_used] = models_used.get(result.model_used, 0) + 1
        if result.provider_used:
            providers_used[result.provider_used] = providers_used.get(result.provider_used, 0) + 1
    
    # Error analysis
    error_types = {}
    status_codes = {}
    for result in failed_results:
        if result.error:
            error_types[result.error] = error_types.get(result.error, 0) + 1
        if result.status_code:
            status_codes[result.status_code] = status_codes.get(result.status_code, 0) + 1
    
    # Token usage
    total_tokens = sum(r.tokens_used for r in successful_results)
    avg_tokens = total_tokens / len(successful_results) if successful_results else 0
    
    # Calculate statistics
    stats = TestStatistics(
        total_requests=num_requests,
        successful_requests=len(successful_results),
        failed_requests=len(failed_results),
        success_rate=(len(successful_results) / num_requests) * 100,
        total_test_time_ms=total_test_time_ms,
        
        avg_response_time_ms=mean(response_times),
        median_response_time_ms=median(response_times),
        min_response_time_ms=min(response_times),
        max_response_time_ms=max(response_times),
        std_dev_response_time_ms=stdev(response_times) if len(response_times) > 1 else 0,
        
        requests_per_second=num_requests / (total_test_time_ms / 1000),
        
        models_used=models_used,
        providers_used=providers_used,
        
        error_types=error_types,
        status_codes=status_codes,
        
        total_tokens=total_tokens,
        avg_tokens_per_request=avg_tokens
    )
    
    return stats


def print_statistics(stats: TestStatistics):
    """Print detailed statistics in a formatted way"""
    
    print("\n" + "=" * 80)
    print("LOAD TEST RESULTS - LOW INTELLIGENCE REQUESTS")
    print("=" * 80)
    
    # Overall Results
    print(f"\n📊 OVERALL RESULTS:")
    print(f"  Total Requests:      {stats.total_requests}")
    print(f"  Successful:          {stats.successful_requests}")
    print(f"  Failed:              {stats.failed_requests}")
    print(f"  Success Rate:        {stats.success_rate:.1f}%")
    print(f"  Total Test Time:     {stats.total_test_time_ms:.0f} ms ({stats.total_test_time_ms/1000:.2f}s)")
    
    # Performance Metrics
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"  Requests/Second:     {stats.requests_per_second:.2f}")
    print(f"  Avg Response Time:   {stats.avg_response_time_ms:.0f} ms")
    print(f"  Median Response:     {stats.median_response_time_ms:.0f} ms") 
    print(f"  Min Response Time:   {stats.min_response_time_ms:.0f} ms")
    print(f"  Max Response Time:   {stats.max_response_time_ms:.0f} ms")
    print(f"  Std Deviation:       {stats.std_dev_response_time_ms:.0f} ms")
    
    # Model/Provider Usage
    if stats.models_used:
        print(f"\n🤖 MODELS USED:")
        for model, count in sorted(stats.models_used.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats.successful_requests) * 100
            print(f"  {model:20} {count:3d} requests ({percentage:5.1f}%)")
    
    if stats.providers_used:
        print(f"\n🔌 PROVIDERS USED:")
        for provider, count in sorted(stats.providers_used.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats.successful_requests) * 100
            print(f"  {provider:20} {count:3d} requests ({percentage:5.1f}%)")
    
    # Token Usage
    if stats.total_tokens > 0:
        print(f"\n🪙 TOKEN USAGE:")
        print(f"  Total Tokens:        {stats.total_tokens:,}")
        print(f"  Avg Tokens/Request:  {stats.avg_tokens_per_request:.1f}")
    
    # Error Analysis (if any failures)
    if stats.failed_requests > 0:
        print(f"\n❌ ERROR ANALYSIS:")
        if stats.error_types:
            for error, count in sorted(stats.error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error[:50]:50} {count} times")
        
        if stats.status_codes:
            print(f"\n📊 HTTP STATUS CODES:")
            for code, count in sorted(stats.status_codes.items()):
                print(f"  HTTP {code:3d}:             {count} requests")
    
    print("\n" + "=" * 80)


async def main():
    """Main entry point"""
    
    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:10006/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Server is healthy: {health_data.get('status', 'unknown')}")
                else:
                    print(f"❌ Server health check failed with status {response.status}")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the bkvy is running on http://localhost:10006")
        return
    
    # Run the load test
    stats = await run_load_test(num_requests=50)
    
    # Print results
    print_statistics(stats)
    
    # Save results to JSON file
    results_file = f"load_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(asdict(stats), f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
