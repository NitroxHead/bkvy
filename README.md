# bkvy

I've grown tired of reimplementing robust LLM API management for every application. Existing solutions are either too complex for my needs or don't address my specific requirements. Most of my LLM interactions happen within preset contexts and is model-agnostic, also I know beforehand the intelligence level each request requires - whether it needs a simple, fast model for basic tasks or a sophisticated model for complex reasoning. These requirements led me to create bkvy. It addresses these challenges:

1. **💰 Cost Optimization**: Automatically routes simple requests to cost-effective models while reserving premium models for complex tasks

2. **🔄 Reliability**: Built-in failover across multiple providers (OpenAI, Anthropic, Google) with retry logic and rate limit management

3. **🎯 Intelligence Matching**: Three-tier system (low/medium/high) ensures optimal model selection for task complexity

4. **🧩 Thinking Control**: Automatically disables "thinking" tokens for simple tasks, reducing costs and latency

5. **📝 Scenario Based Fallback**: Based on predefined scenarios it seeks answers.

## ⚙️ Core Architecture

### Intelligence-Based Routing

- **Low Intelligence**: Simple questions, translations, basic tasks → Gemini Flash, Claude Haiku
- **Medium Intelligence**: Analysis, coding, complex reasoning → GPT-4o Mini, Claude Sonnet  
- **High Intelligence**: Research, advanced reasoning, critical tasks → GPT-4o, Claude Opus

### Advanced Features

- **Circuit Breaker**: Production-grade failure isolation with shared key awareness
- **Global Timeout**: Two-level timeout (soft 30s, hard 120s) prevents indefinite loops
- **Multi-Provider Failover**: Seamless switching between OpenAI, Anthropic, Google, Ollama
- **Rate Limit Management**: Per-key, per-model tracking with automatic backoff
- **Queue Optimization**: Intelligent request scheduling and load balancing
- **Background Health Probing**: Automatic recovery detection with minimal cost
- **Anti-Flapping Protection**: Detects and mitigates unstable circuits
- **Scenario-Based Routing**: Pre-configured routing strategies for different use cases
- **Thinking Control**: Automatic token optimization for reasoning models
- **Optional Statistics Logging**: Track usage patterns without storing message content

### Project Structure

```
bkvy/
├── models/                  # Data models and schemas
│   ├── enums.py            # IntelligenceLevel, RoutingMethod
│   ├── data_classes.py     # RateLimitState, QueueState, etc.
│   └── schemas.py          # Pydantic request/response models
├── core/                   # Core business logic
│   ├── config.py           # Configuration management
│   ├── rate_limits.py      # Rate limiting functionality
│   ├── queues.py           # Queue management
│   ├── llm_client.py       # LLM API client
│   └── router.py           # Intelligent routing engine
├── api/                    # FastAPI application
│   ├── app.py              # FastAPI endpoints
│   ├── lifespan.py         # Application lifecycle management
│   ├── middleware.py       # IP whitelist middleware for dashboard
│   └── templates/          # HTML templates
│       └── dashboard.html  # Dashboard UI
└── utils/                  # Utility functions
    ├── logging.py          # Structured logging setup
    ├── transaction_logger.py  # Detailed CSV transaction logs
    ├── summary_stats.py    # Lightweight daily statistics
    └── dashboard.py        # Dashboard data processing

main.py                     # Application entry point
config/                     # Configuration files
├── providers.json          # API keys and model configs
└── routing.json            # Scenario definitions
docs/                       # Documentation
├── API.md                  # API reference
├── CONFIGURATION.md        # Configuration guide
├── DASHBOARD.md            # Dashboard documentation
├── EXAMPLES.md             # Usage examples
└── SETUP.md                # Setup instructions
tests/                      # Test files
├── test_dashboard_sample_data.py
├── test_transaction_logging.py
└── test_load_low_intelligence.py
logs/                       # Optional logging outputs (runtime)
├── transactions.csv        # Detailed request logs (if enabled)
└── daily_stats.json        # Daily aggregated statistics (if enabled)
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/NitroxHead/bkvy
cd bkvy
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example configurations
cp config/providers.example.json config/providers.json
cp config/routing.example.json config/routing.json

# Add your API keys
nano config/providers.json
```

### 3. Start the Server

```bash
python main.py
```

### 4. Make Your First Request

**Simple Request (Auto-routes to cheapest model):**

```bash
curl -X POST "http://localhost:10006/llm/intelligence" \\
  -H "Content-Type: application/json" \\
  -d '{
    "client_id": "demo-client",
    "intelligence_level": "low",
    "max_wait_seconds": 30,
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

**Complex Request (Auto-routes to premium model):**

```bash
curl -X POST "http://localhost:10006/llm/intelligence" \\
  -H "Content-Type: application/json" \\
  -d '{
    "client_id": "demo-client", 
    "intelligence_level": "high",
    "max_wait_seconds": 60,
    "messages": [{"role": "user", "content": "Analyze the economic implications of quantum computing"}]
  }'
```

**max_wait_seconds** is a real per-request time budget: no new attempts start once it is spent (always capped by `REQUEST_HARD_TIMEOUT_SECONDS`, default 120s). A value <= 0 means no per-request override.

## 🎛️ Routing Modes

### 1. Intelligence-Based Routing

Automatically selects optimal models based on task complexity:

```json
{
  "intelligence_level": "low|medium|high",
  "messages": [...],
  "debug": false
}
```

### 2. Scenario-Based Routing

Use pre-configured routing strategies:

```json
{
  "scenario": "cost_optimized_bulk",
  "messages": [...] 
}
```

### 3. Direct Routing

Route to specific provider/model combinations:

```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "messages": [...]
}
```

## 🔧 Configuration

### Providers Configuration (`config/providers.json`)

```json
{
  "openai": {
    "keys": {
      "openai_key_1": {
        "api_key": "sk-proj-...",
        "rate_limits": {
          "gpt-4o": {"rpm": 10, "rpd": 200}
        }
      }
    },
    "models": {
      "gpt-4o": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "cost_per_1k_tokens": 0.0025,
        "intelligence_tier": "high",
        "supports_thinking": false
      }
    }
  }
}
```

### Routing Scenarios (`config/routing.json`)

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

## 📡 API Reference

### Endpoints

- `POST /llm/intelligence` - Intelligence-based routing
- `POST /llm/scenario` - Scenario-based routing  
- `POST /llm/direct` - Direct provider routing
- `GET /health` - Health check
- `GET /providers` - Provider status
- `GET /intelligence/{level}` - Available models by intelligence

### Response Formats

**Simplified (default):**

```json
{
  "success": true,
  "request_id": "uuid",
  "model_used": "gemini-2.5-flash",
  "content": "The answer is 4",
  "usage": {"input_tokens": 4, "output_tokens": 5},
  "finish_reason": "stop"
}
```

**Debug mode (`debug: true`):**

```json
{
  "success": true,
  "provider_used": "gemini",
  "model_used": "gemini-2.5-flash", 
  "routing_method": "intelligence",
  "decision_reason": "cheapest_within_estimate",
  "metadata": {
    "alternatives_considered": [...],
    "total_completion_time_ms": 1250
  }
}
```

## 🔍 Monitoring & Debugging

### Health Check

```bash
curl http://localhost:10006/health
```

### Provider Status

```bash
curl http://localhost:10006/providers
```

### Queue Status

```bash
curl http://localhost:10006/queue/status
```

### Rate Limit Status

```bash
curl http://localhost:10006/rates/status
```

### Statistics (if enabled)

```bash
# Check logging status
curl http://localhost:10006/statistics/status

# Get aggregate statistics
curl http://localhost:10006/statistics/aggregate

# Get daily statistics
curl http://localhost:10006/statistics/daily/2025-01-15
```

## 📊 Statistics & Logging

bkvy includes **two optional, independent logging systems** that track usage patterns while respecting privacy (no message content is logged).

### Transaction Logging (Detailed CSV)

Tracks full request details for debugging and auditing.

**Enable:**
```bash
# In bkvy.service or environment
export TRANSACTION_LOGGING=true
```

**Output:** `logs/transactions.csv`

**Includes:**
- Request metadata (client_id, routing_method, intelligence_level)
- Performance metrics (response time, tokens, cost estimates)
- Routing decisions (provider, model, API key used)
- Fallback attempts and alternatives tried
- Error types and messages

**Use cases:** Detailed analysis, debugging, compliance auditing

### Summary Statistics (Lightweight JSON)

Maintains daily aggregated statistics independent of detailed logs.

**Enable:**
```bash
# In bkvy.service or environment
export SUMMARY_STATS=true
```

**Output:** `logs/daily_stats.json`

**Includes (by day):**
- Total requests and success rates
- Breakdown by routing method, intelligence level, provider
- Error counts by type
- Total cost estimates
- Average response times

**Use cases:** Monitoring dashboards, trend analysis, capacity planning

**Example daily_stats.json:**
```json
{
  "2025-01-15": {
    "total_requests": 1247,
    "successful": 1198,
    "by_routing_method": {"intelligence": 856, "scenario": 298, "direct": 93},
    "by_intelligence": {"low": 623, "medium": 412, "high": 212},
    "by_provider": {"openai": 534, "anthropic": 421, "gemini": 243},
    "errors": {"rate_limited": 35, "model_not_found": 8},
    "total_cost": 23.46,
    "avg_response_time_ms": 1250
  }
}
```

### Privacy & Security

- ✅ **No message content** stored in either logging system
- ✅ **Disabled by default** - opt-in only
- ✅ **API keys logged as identifiers** only (e.g., "openai_key_1"), not actual values
- ✅ **Local storage** - all data stays on your server
- ✅ **Independent systems** - use one, both, or neither

### Statistics API Endpoints

- `GET /statistics/status` - Check which logging systems are enabled
- `GET /statistics/summary` - Transaction log summary (if enabled)
- `GET /statistics/aggregate` - Aggregated stats across all days
- `GET /statistics/daily/{date}` - Stats for specific date (YYYY-MM-DD)
- `GET /statistics/daily` - All daily statistics (pivot table)

## 📊 Web Dashboard

bkvy includes an **optional browser-based dashboard** for visualizing transaction logs and system statistics with real-time charts and analytics.

### Quick Start

```bash
# Enable the dashboard
export DASHBOARD_ENABLED=true
export TRANSACTION_LOGGING=true  # Required for dashboard data

# Optional: Configure IP access control
export DASHBOARD_ALLOWED_IPS="127.0.0.1"  # Local only (default)

# Access the dashboard
# http://localhost:10006/dashboard
```

### Key Features

- **📈 Real-time Statistics**: Requests, success rates, costs, response times (P95/P99)
- **📊 Interactive Charts**: Time-series, provider/model distributions, wait time analysis
- **📝 Transaction History**: Recent requests with full details
- **🔑 API Key Analytics**: Per-key usage, costs, and performance metrics
- **⚠️ Error Analysis**: Error tracking and failure patterns
- **🎨 Dark/Light Mode**: Automatic theme detection
- **🔄 Auto-refresh**: 15s, 30s, 1m, 5m intervals
- **📅 Custom Date Ranges**: Flexible time range selection (1h - 365d)
- **🔒 IP Access Control**: Whitelist-based security

**For complete documentation, see [docs/DASHBOARD.md](docs/DASHBOARD.md)**

## 🛠️ Advanced Features

### Thinking Control

Automatically optimizes thinking tokens for reasoning models:

```python
# Low intelligence requests automatically disable thinking
{
  "intelligence_level": "low",  # Auto-sets thinking_budget=0 for Gemini
  "messages": [{"role": "user", "content": "Simple question"}]
}

# Manual thinking control
{
  "options": {
    "disable_thinking": true  # Explicit control
  }
}
```

### Custom Scenarios

Create application-specific routing strategies:

```json
{
  "scenarios": {
    "customer_support": [
      {"provider": "anthropic", "model": "claude-3-haiku", "priority": 1},
      {"provider": "openai", "model": "gpt-4o-mini", "priority": 2}
    ],
    "code_generation": [
      {"provider": "anthropic", "model": "claude-3-sonnet", "priority": 1},
      {"provider": "openai", "model": "gpt-4o", "priority": 2}
    ]
  }
}
```

## ⚡ Circuit Breaker & Reliability

bkvy includes a **production-grade circuit breaker** that prevents cascade failures, handles shared API keys correctly, and enforces global request timeouts.

### What is a Circuit Breaker?

Traditional retry logic wastes time on failing providers. A circuit breaker **learns from failures** and temporarily blocks unhealthy providers, dramatically reducing latency when things go wrong.

**Without Circuit Breaker:**
```
Request → Try Provider A (30s timeout) → Fail
       → Try Provider A again (30s) → Fail
       → Try Provider A again (30s) → Fail
       → Try Provider B → Success
Total: 90+ seconds
```

**With Circuit Breaker:**
```
Request → Provider A circuit is OPEN (instant skip)
       → Try Provider B → Success
Total: 2 seconds
```

### Key Features

#### 🔌 **Intelligent Circuit States**
- **CLOSED** (Healthy): Normal operation
- **OPEN** (Failed): Block requests, skip instantly
- **HALF_OPEN** (Testing): Probe for recovery

#### 📊 **Dual Failure Threshold**
Circuits open when either condition is met:
1. **Consecutive failures**: 3 failures in a row (immediate pattern detection)
2. **Sliding window**: 5 failures within 10 minutes (sustained issues)

This catches both **sudden failures** (3 consecutive) and **intermittent problems** (spread over time).

#### 🎯 **Shared Key Awareness**
Traditional circuit breakers assume you control rate limits. But **API keys can be shared externally**:

```bash
# Your app uses 5 RPM, external app uses 8 RPM
# Traditional: Assumes quota reset → immediate 429
# bkvy: Always tests before assuming recovery ✅
```

#### ⏱️ **Global Request Timeout**
Two-level timeout prevents indefinite loops:
- **Soft timeout (30s)**: Escalate to fast mode
  - Reduce retries: 3 → 1
  - Shorter API timeouts: 300s → 30s
  - Only use CLOSED circuits
  - Prioritize speed over cost
- **Hard timeout (120s)**: Abort completely

#### 🔍 **Intelligent Failure Classification**

| Error Type | Strategy | Backoff | Auto-Recovery |
|------------|----------|---------|---------------|
| **429 Rate Limit** | Skip alternative, test before reopening | Calculated from headers (max 24h) | ✅ Yes (with probe) |
| **5xx Service Error** | 3 retries with backoff | Exponential (max 30min) | ✅ Yes (with probe) |
| **401/403 Auth Error** | Skip entire provider | Infinite | ❌ Requires manual fix |
| **Timeout** | 5 retries (lenient) | Exponential (max 5min) | ✅ Yes (with probe) |
| **Content Error** | Request-specific, 1 retry | No backoff | ✅ Yes |

#### 🚨 **Anti-Flapping Protection**
Detects rapid open/close cycles (3+ in 5 minutes):
- Applies 5x backoff penalty
- Deprioritizes unstable circuits
- Clears after 10 minutes of stability

#### 🏥 **Background Health Probing**
- Runs every 10 seconds
- Uses minimal-cost probes:
  - **Ollama**: `/api/version` (free)
  - **Gemini**: Model info endpoint (free)
  - **OpenAI/Anthropic**: max_tokens=1
- Max 5 concurrent probes (prevents thundering herd)

### Configuration

```bash
# Circuit Breaker
export CIRCUIT_BREAKER_ENABLED=true          # Enable/disable
export CIRCUIT_FAILURE_THRESHOLD=3           # Consecutive failures before opening
export CIRCUIT_SLIDING_WINDOW_SECONDS=600    # Sliding window duration (10 min)
export CIRCUIT_SLIDING_WINDOW_THRESHOLD=5    # Failures in window to open
export CIRCUIT_MAX_BACKOFF_SECONDS=1800      # Max 30 minutes
export CIRCUIT_FLAPPING_THRESHOLD=3          # Opens in 5min = flapping

# Health Probing
export HEALTH_PROBE_ENABLED=true             # Background probing
export HEALTH_PROBE_INTERVAL_SECONDS=10      # Probe every 10s
export HEALTH_PROBE_TIMEOUT_SECONDS=10       # Probe timeout

# Global Timeouts
export REQUEST_SOFT_TIMEOUT_SECONDS=30       # Escalation threshold
export REQUEST_HARD_TIMEOUT_SECONDS=120      # Absolute maximum
```

### Monitoring Endpoints

```bash
# Check all circuit states
curl http://localhost:10006/circuits/status

# Check specific provider health
curl http://localhost:10006/circuits/provider/gemini

# Get summary of all providers
curl http://localhost:10006/circuits/summary

# Check specific circuit
curl http://localhost:10006/circuits/gemini/gemini-2.5-flash/gemini_key_1

# Manually reset circuit (admin action)
curl -X POST http://localhost:10006/circuits/reset/gemini/gemini-2.5-flash/gemini_key_1
```

### Real-World Scenarios

#### Scenario 1: Rate Limit Hit (Shared Key)
```
T=0:00  Your app hits 429
        → Circuit opens immediately
        → Next test time = T+1:00 (from response headers)

T=0:30  External app still using key (unknown to you)

T=1:00  Background worker probes
        → Still 429 (external usage)
        → Circuit stays OPEN, backoff increases to T+2:00

T=2:00  Background worker probes again
        → Success! External usage stopped
        → Circuit closes
        → Normal operation resumes
```

#### Scenario 2: Service Outage
```
3 consecutive 503 errors
→ Circuit opens
→ Exponential backoff: 30s, 60s, 120s...
→ Background worker probes at intervals
→ On success: Circuit closes
```

#### Scenario 3: Auth Failure
```
Single 401 error
→ Circuit opens
→ All circuits for provider skip
→ Never auto-recovers (requires config fix)
→ Logged as CRITICAL alert
```

#### Scenario 4: Flapping Circuit
```
Opens 3+ times in 5 minutes
→ Flapping detected
→ Backoff multiplied by 5
→ Circuit deprioritized
→ Clears after 10 minutes of stability
```

### Testing

Run the test suite:

```bash
# Start server with circuit breaker enabled
export CIRCUIT_BREAKER_ENABLED=true
python3 main.py

# In another terminal, run tests
python3 test_circuit_breaker.py
```

### Performance Impact

**Benefits:**
- ⚡ **Latency reduction**: 30s+ → 2-5s when providers fail
- 💰 **API cost savings**: 20-30% (fewer wasted retries)
- 🚀 **Faster recovery**: Proactive health checks
- 🛡️ **Prevents cascades**: Circuit isolation

**Overhead:**
- Memory: ~1KB per circuit state
- Disk: ~2KB per persisted circuit (JSON)
- CPU: Minimal (background worker every 10s)
- Probe cost: Minimal (free endpoints for Ollama/Gemini)

### Advanced: Provider-Level Health

Circuit breaker aggregates individual circuits into provider-level health:

```json
{
  "provider_name": "gemini",
  "overall_health": "degraded",
  "total_circuits": 12,
  "open_circuits": 3,
  "failure_pattern": "rate_limited",
  "recommended_action": "external_usage_likely",
  "should_skip_provider": false
}
```

**Detected Patterns:**
- `auth` - All circuits auth failures → Check API keys
- `rate_limited` - All circuits 429 → External usage
- `widespread` - 80%+ circuits open → Provider outage
- `isolated` - Few circuits open → Specific keys/models affected

## 📈 Performance Tuning

### Rate Limit Optimization

- Configure `rpm` (requests per minute) and `rpd` (requests per day) per model
- System automatically manages queues and backoff
- Circuit breaker respects rate limit windows

### Queue Management

- Concurrent request processing
- Intelligent wait time estimation
- Automatic load balancing across API keys

### Cost Optimization

- Set appropriate `cost_per_1k_tokens` values
- Use intelligence tiers to prevent over-spending
- Monitor usage with built-in metrics
- Circuit breaker prevents wasted API calls on failing providers

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│ Intelligent     │────│ Configuration   │
│                 │    │ Router          │    │ Manager         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Rate Limit      │────│ Queue           │────│ LLM Client      │
│ Manager         │    │ Manager         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ OpenAI API      │    │ Anthropic API   │    │ Google AI API   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤝 Contributing

I welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🆘 Support

- 📖 [Documentation](docs/)
- 🐛 [Issues](https://github.com/NitroxHead/bkvy/issues)

## 🔮 Roadmap

- [ ] Custom model fine-tuning routing
- [ ] Docker implementation
- [ ] Prompt caching optimization
