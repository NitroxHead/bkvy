# Sliding Window Failure Tracking Implementation

## Overview

The circuit breaker now uses a **proper sliding window** for failure tracking instead of simple counters. This provides more accurate failure detection by tracking individual failure events with timestamps.

## What Changed

### 1. Data Model (`bkvy/models/circuit_states.py`)

#### New Data Structure
```python
@dataclass
class FailureEvent:
    """Record of a single failure occurrence"""
    timestamp: datetime
    failure_type: FailureType
    error_message: str
    response_time_ms: float = 0.0
```

#### Updated CircuitState
- **Added**: `failure_history: List[FailureEvent]` - Stores individual failures
- **Changed**: `failure_count` is now a computed property from the sliding window
- **Auto-cleanup**: Old failures outside the window are automatically removed

### 2. New Methods

#### `add_failure_to_window()`
```python
circuit.add_failure_to_window(
    failure_type=FailureType.RATE_LIMIT_429,
    error_message="Rate limit exceeded",
    response_time_ms=250.0,
    window_seconds=600  # 10 minutes
)
```
- Adds failure to history
- Auto-cleans entries outside window
- Updates computed `failure_count`

#### `get_failure_count_in_window()`
```python
count = circuit.get_failure_count_in_window(window_seconds=600)
```
- Returns count of failures in the specified time window
- Always accurate because it counts from timestamps

#### `get_failures_by_type_in_window()`
```python
rate_limit_count = circuit.get_failures_by_type_in_window(
    failure_type=FailureType.RATE_LIMIT_429,
    window_seconds=600
)
```
- Filter failures by type within window
- Useful for pattern detection

#### `clear_failure_window()`
```python
circuit.clear_failure_window()
```
- Clears all failure history
- Called on successful recovery

### 3. Dual Threshold Logic (`bkvy/core/circuit_breaker.py`)

Circuits now open when **either** condition is met:

```python
if strategy.should_circuit_break:
    # Threshold 1: Consecutive failures (immediate pattern)
    if circuit.consecutive_failures >= self.failure_threshold:  # 3
        should_open = True

    # Threshold 2: Sliding window (sustained issues)
    elif circuit.get_failure_count_in_window(self.sliding_window_seconds) >= self.sliding_window_threshold:  # 5 in 10min
        should_open = True
```

## Configuration

### New Environment Variables

```bash
# Sliding window duration (default: 600 seconds = 10 minutes)
export CIRCUIT_SLIDING_WINDOW_SECONDS=600

# Number of failures in window to open circuit (default: 5)
export CIRCUIT_SLIDING_WINDOW_THRESHOLD=5
```

### Existing Variables Still Work

```bash
# Consecutive failures before opening (default: 3)
export CIRCUIT_FAILURE_THRESHOLD=3
```

## Benefits

### 1. **Detects Intermittent Failures**
Old approach: Only counted consecutive failures
```
Success → Fail → Success → Fail → Success → Fail → Success
         ❌ Circuit stays CLOSED (no 3 consecutive)
```

New approach: Sliding window catches pattern
```
Success → Fail → Success → Fail → Success → Fail → Success
         ✅ Circuit OPENS (3 failures in 10 minutes)
```

### 2. **Auto-Expiring History**
Old approach: Manual decay, could be stale
```python
circuit.failure_count = max(0, circuit.failure_count - 1)  # Linear decay
```

New approach: Time-based automatic expiry
```python
# Failures from 11 minutes ago automatically don't count
count = circuit.get_failure_count_in_window(600)
```

### 3. **More Accurate Tracking**
- Each failure stores timestamp, type, error message, response time
- Can analyze failure patterns over time
- Better debugging with full failure history

### 4. **Better Pattern Detection**
```python
# Check if recent failures are all rate limits
rate_limit_failures = circuit.get_failures_by_type_in_window(
    FailureType.RATE_LIMIT_429,
    window_seconds=600
)
if rate_limit_failures >= 3:
    # Likely external API key usage
```

## Example Scenarios

### Scenario 1: Sudden Outage
```
T=0:00  → Fail (503)
T=0:01  → Fail (503)
T=0:02  → Fail (503)
         ✅ Circuit OPENS (3 consecutive failures)
```

### Scenario 2: Intermittent Issues
```
T=0:00  → Success
T=0:02  → Fail (429)
T=0:05  → Success
T=0:07  → Fail (429)
T=0:10  → Success
T=0:12  → Fail (503)
T=0:15  → Success
T=0:18  → Fail (timeout)
T=0:20  → Success
T=0:22  → Fail (503)

Consecutive: Only 1 (no pattern)
Sliding window: 5 failures in last 10 minutes
         ✅ Circuit OPENS (sliding window threshold)
```

### Scenario 3: Old Failures Expire
```
T=0:00  → Fail (stored in window)
T=0:05  → Fail (stored in window)
T=0:10  → Fail (stored in window)
         failure_count = 3

T=10:01 → get_failure_count_in_window(600)
         → Failures at 0:00, 0:05, 0:10 are all >10min old
         → failure_count = 0 (auto-expired!)
```

## Serialization

Failure history is fully persisted to disk:

```json
{
  "combination_key": "gemini_flash_key1",
  "failure_history": [
    {
      "timestamp": "2025-10-13T12:34:56.789Z",
      "failure_type": "rate_limit_429",
      "error_message": "Rate limit exceeded",
      "response_time_ms": 250.0
    }
  ],
  "failure_count": 1
}
```

## Performance Impact

- **Memory**: ~100 bytes per failure event
- **Typical**: 5-10 failures in window = ~1KB per circuit
- **Cleanup**: Automatic on every new failure
- **Persistence**: Saved with circuit state (atomic writes)

## Migration

No migration needed! The implementation is backward compatible:

1. Old circuit states without `failure_history` will initialize with empty list
2. `failure_count` field is preserved for compatibility
3. New failures automatically use sliding window
4. Existing circuits continue working normally

## Testing

The sliding window can be tested with:

```bash
# Start server
export CIRCUIT_BREAKER_ENABLED=true
export CIRCUIT_SLIDING_WINDOW_SECONDS=60  # 1 minute for faster testing
export CIRCUIT_SLIDING_WINDOW_THRESHOLD=3
python3 main.py

# Send 3 spaced requests (not consecutive)
# Wait 20s between each
# After 3rd failure, circuit should OPEN due to sliding window
```

## Summary

✅ **Implemented**: Full sliding window with time-based expiry
✅ **Dual threshold**: 3 consecutive OR 5 in 10 minutes
✅ **Auto-cleanup**: Old failures expire automatically
✅ **Persisted**: Full failure history saved to disk
✅ **Configurable**: Window size and threshold via env vars
✅ **Backward compatible**: Works with existing circuit states
✅ **Documented**: README and implementation docs updated
