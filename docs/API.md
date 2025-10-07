# API Documentation

Complete API reference for the bkvy.

## Base URL

```
http://localhost:10006
```

## Authentication

No authentication required for the API endpoints. Security should be handled at the infrastructure level.

## Core Routing Endpoints

### Intelligence-Based Routing

Route requests based on task complexity.

**Endpoint:** `POST /llm/intelligence`

**Request Body:**
```json
{
  "client_id": "string",
  "intelligence_level": "low|medium|high",
  "max_wait_seconds": 30,
  "messages": [
    {"role": "user|assistant|system", "content": "string"}
  ],
  "options": {
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 40,
    "stop": ["stop_sequence"],
    "disable_thinking": false
  },
  "debug": false
}
```

**Parameters:**
- `client_id` (required): Unique identifier for the client
- `intelligence_level` (required): Task complexity level
  - `low`: Simple questions, basic tasks
  - `medium`: Analysis, coding, reasoning  
  - `high`: Complex research, advanced reasoning
- `max_wait_seconds` (required): Maximum acceptable wait time
- `messages` (required): Conversation history
- `options` (optional): Model generation parameters
- `debug` (optional): Return detailed debug information (default: false)

**Response (Simplified):**
```json
{
  "success": true,
  "request_id": "uuid",
  "model_used": "gemini-2.5-flash",
  "content": "Response content",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 25,
    "total_tokens": 35
  },
  "finish_reason": "stop",
  "truncated": false,
  "error": null
}
```

**Response (Debug Mode):**
```json
{
  "success": true,
  "request_id": "uuid",
  "provider_used": "gemini",
  "model_used": "gemini-2.5-flash",
  "api_key_used": "gemini_key_1",
  "routing_method": "intelligence",
  "decision_reason": "cheapest_within_estimate_after_1_attempts",
  "response": {
    "content": "Response content",
    "usage": {"input_tokens": 10, "output_tokens": 25, "total_tokens": 35},
    "finish_reason": "stop",
    "truncated": false,
    "raw_response": {...}
  },
  "metadata": {
    "rate_limit_wait_ms": 0,
    "queue_wait_ms": 0,
    "api_response_time_ms": 1250,
    "total_completion_time_ms": 1300,
    "cost_usd": 0.0,
    "alternatives_considered": [...]
  },
  "error_code": null,
  "message": null
}
```

### Scenario-Based Routing

Route using predefined scenarios.

**Endpoint:** `POST /llm/scenario`

**Request Body:**
```json
{
  "client_id": "string",
  "scenario": "scenario_name",
  "max_wait_seconds": 30,
  "messages": [...],
  "options": {...},
  "debug": false
}
```

**Available Scenarios:**
- `low_intelligence_low_rate`: Cost-optimized for simple tasks
- `medium_intelligence_balanced`: Balanced performance/cost
- `high_intelligence_premium`: Premium models for complex tasks
- `reasoning_models`: For tasks requiring deep reasoning
- `cost_optimized_bulk`: Cheapest options for bulk processing
- `gemini_only_fast`: Gemini models only
- `anthropic_preferred`: Anthropic models preferred

### Direct Routing

Route to specific provider/model combination.

**Endpoint:** `POST /llm/direct`

**Request Body:**
```json
{
  "client_id": "string",
  "provider": "openai|anthropic|gemini",
  "model": "model_name",
  "api_key_id": "optional_key_id",
  "max_wait_seconds": 30,
  "messages": [...],
  "options": {...},
  "debug": false
}
```

**Parameters:**
- `provider` (required): Provider name
- `model` (required): Model name (alias for `model_name`)
- `api_key_id` (optional): Specific API key to use (auto-select if not provided)

## Monitoring Endpoints

### Health Check

Check system health.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "mode": "synchronous"
}
```

### Provider Status

Get status of all providers and models.

**Endpoint:** `GET /providers`

**Response:**
```json
{
  "providers": {
    "openai": {
      "models": {
        "gpt-4o": {
          "intelligence_tier": "high",
          "cost_per_1k_tokens": 0.0025,
          "avg_response_time_ms": 4500,
          "keys": {
            "openai_key_1": {
              "rate_limits": {"rpm": 10, "rpd": 200},
              "currently_limited": false,
              "wait_time_seconds": 0,
              "queue_wait_seconds": 0
            }
          }
        }
      }
    }
  }
}
```

### Intelligence Level Status

Get available models for a specific intelligence level.

**Endpoint:** `GET /intelligence/{level}`

**Parameters:**
- `level`: `low`, `medium`, or `high`

**Response:**
```json
{
  "intelligence_level": "low",
  "models": [
    {
      "provider": "gemini",
      "model": "gemini-2.5-flash",
      "intelligence_tier": "low",
      "cost_per_1k_tokens": 0.0,
      "avg_response_time_ms": 2000,
      "keys_status": [
        {
          "api_key_id": "gemini_key_1",
          "rate_limited": false,
          "rate_limit_wait_seconds": 0,
          "queue_wait_seconds": 0,
          "total_wait_seconds": 0
        }
      ]
    }
  ]
}
```

### Available Scenarios

List all configured scenarios.

**Endpoint:** `GET /scenarios`

**Response:**
```json
{
  "scenarios": {
    "cost_optimized": [
      {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1},
      {"provider": "openai", "model": "gpt-4o-mini", "priority": 2}
    ]
  }
}
```

### Queue Status

Get current queue states.

**Endpoint:** `GET /queue/status`

**Response:**
```json
{
  "queue_states": {
    "gemini_gemini-2.5-flash_gemini_key_1": {
      "current_queue_length": 0,
      "active_requests": [],
      "estimated_queue_wait_seconds": 0.0,
      "last_updated": "2024-01-15T10:30:00Z"
    }
  }
}
```

### Rate Limit Status

Get current rate limit states.

**Endpoint:** `GET /rates/status`

**Response:**
```json
{
  "rate_limit_states": {
    "gemini_gemini-2.5-flash_gemini_key_1": {
      "requests_this_minute": 0,
      "requests_today": 5,
      "currently_rate_limited": false,
      "rate_limit_wait_seconds": 0.0,
      "rpm_limit": 10,
      "rpd_limit": 250,
      "minute_reset_time": "2024-01-15T10:31:00Z",
      "day_reset_time": "2024-01-16T00:00:00Z"
    }
  }
}
```

## Statistics Endpoints

Optional logging systems that track usage patterns while respecting privacy (no message content is stored).

### Statistics Status

Check which logging systems are enabled.

**Endpoint:** `GET /statistics/status`

**Response:**
```json
{
  "transaction_logging": {
    "enabled": true,
    "log_file": "/path/to/logs/transactions.csv",
    "log_directory": "/path/to/logs"
  },
  "summary_stats": {
    "enabled": true,
    "stats_file": "/path/to/logs/daily_stats.json",
    "log_directory": "/path/to/logs"
  }
}
```

### Transaction Log Summary

Get aggregate statistics from detailed transaction logs (CSV).

**Endpoint:** `GET /statistics/summary`

**Requires:** `TRANSACTION_LOGGING=true`

**Response:**
```json
{
  "enabled": true,
  "total_requests": 1247,
  "successful_requests": 1198,
  "success_rate": 0.961,
  "routing_methods": {
    "intelligence": 856,
    "scenario": 298,
    "direct": 93
  },
  "providers_used": {
    "openai": 534,
    "anthropic": 421,
    "gemini": 243
  },
  "intelligence_levels": {
    "low": 623,
    "medium": 412,
    "high": 212
  },
  "errors": {
    "rate_limited": 35,
    "model_not_found": 8
  },
  "total_cost_estimate": 23.456789,
  "log_file": "/path/to/logs/transactions.csv"
}
```

### Aggregate Statistics

Get aggregated statistics across all days from summary stats.

**Endpoint:** `GET /statistics/aggregate`

**Requires:** `SUMMARY_STATS=true`

**Response:**
```json
{
  "enabled": true,
  "total_requests": 1247,
  "successful_requests": 1198,
  "success_rate": 0.961,
  "routing_methods": {
    "intelligence": 856,
    "scenario": 298,
    "direct": 93
  },
  "providers_used": {
    "openai": 534,
    "anthropic": 421,
    "gemini": 243
  },
  "intelligence_levels": {
    "low": 623,
    "medium": 412,
    "high": 212
  },
  "errors": {
    "rate_limited": 35,
    "model_not_found": 8
  },
  "total_cost_estimate": 23.46,
  "days_tracked": 7,
  "stats_file": "/path/to/logs/daily_stats.json"
}
```

### Daily Statistics

Get statistics for a specific date.

**Endpoint:** `GET /statistics/daily/{date}`

**Requires:** `SUMMARY_STATS=true`

**Parameters:**
- `date`: Date in YYYY-MM-DD format (e.g., `2025-01-15`)

**Response:**
```json
{
  "date": "2025-01-15",
  "total_requests": 178,
  "successful": 171,
  "success_rate": 0.961,
  "by_routing_method": {
    "intelligence": 122,
    "scenario": 43,
    "direct": 13
  },
  "by_intelligence": {
    "low": 89,
    "medium": 59,
    "high": 30
  },
  "by_provider": {
    "openai": 76,
    "anthropic": 60,
    "gemini": 35
  },
  "errors": {
    "rate_limited": 5,
    "model_not_found": 2
  },
  "total_cost": 3.35,
  "avg_response_time_ms": 1250
}
```

### All Daily Statistics

Get statistics for all dates (daily pivot table).

**Endpoint:** `GET /statistics/daily`

**Requires:** `SUMMARY_STATS=true`

**Response:**
```json
{
  "2025-01-15": {
    "total_requests": 178,
    "successful": 171,
    "success_rate": 0.961,
    "by_routing_method": {"intelligence": 122, "scenario": 43, "direct": 13},
    "by_intelligence": {"low": 89, "medium": 59, "high": 30},
    "by_provider": {"openai": 76, "anthropic": 60, "gemini": 35},
    "errors": {"rate_limited": 5},
    "total_cost": 3.35,
    "avg_response_time_ms": 1250
  },
  "2025-01-16": {
    "total_requests": 203,
    "successful": 198,
    "..."
  }
}
```

## Error Responses

All endpoints may return error responses with the following structure:

```json
{
  "success": false,
  "request_id": "uuid",
  "error_code": "error_type",
  "message": "Human readable error message",
  "retry_suggestion": {
    "fastest_available_combination": {
      "provider": "gemini",
      "model": "gemini-2.5-flash",
      "estimated_total_time_seconds": 5.2,
      "available_at": "2024-01-15T10:35:12Z"
    }
  }
}
```

### Common Error Codes

- `no_models_available`: No models match the intelligence level
- `scenario_not_found`: Scenario name not found in configuration
- `provider_not_found`: Provider not configured
- `model_not_found`: Model not found for provider
- `api_key_not_found`: Specific API key not found
- `all_alternatives_failed`: All provider/model combinations failed
- `retry_logic_failure`: Fatal error in retry system
- `no_combinations_available`: No available combinations within time limit

## Model Generation Options

The `options` field supports the following parameters:

### Common Parameters
- `max_tokens` (integer): Maximum tokens to generate
- `temperature` (float 0.0-2.0): Randomness (0=deterministic, higher=more random)
- `top_p` (float 0.0-1.0): Nucleus sampling parameter
- `stop` (array): Stop sequences to end generation

### Provider-Specific Parameters
- `top_k` (integer): Top-k sampling (Anthropic, Gemini)
- `disable_thinking` (boolean): Disable reasoning tokens for supported models

### Thinking Control

For models that support thinking/reasoning tokens:

```json
{
  "options": {
    "disable_thinking": true  // Disables thinking to save tokens/cost
  }
}
```

**Auto-disable behavior:** Low intelligence requests automatically disable thinking for supported models unless explicitly set to `false`.

## Rate Limiting

The router respects the following rate limiting behavior:

1. **Per-key, per-model tracking**: Each API key's rate limits are tracked separately for each model
2. **RPM (Requests Per Minute)**: Short-term rate limiting
3. **RPD (Requests Per Day)**: Long-term quota management
4. **Automatic backoff**: System waits when rate limits are hit
5. **Queue management**: Requests are queued and processed when capacity is available

## Request Flow

1. **Request validation**: Check required parameters
2. **Provider selection**: Choose optimal provider/model combination based on:
   - Intelligence level or scenario requirements
   - Cost optimization
   - Rate limit status
   - Queue wait times
3. **Retry logic**: Automatic failover with exponential backoff
4. **Response formatting**: Return simplified or debug response based on `debug` parameter

