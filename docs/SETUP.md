# Setup Guide

This guide will walk you through setting up the bkvy from scratch.

## Prerequisites

- Python 3.8+
- Git
- API keys for at least one provider (OpenAI, Anthropic, or Google AI)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NitroxHead/bkvy.git
cd bkvy
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration Setup

#### Copy Example Configurations
```bash
cp config/providers.example.json config/providers.json
cp config/routing.example.json config/routing.json
```

#### Configure API Keys

Edit `config/providers.json` to add your API keys:

**OpenAI Configuration:**
```json
{
  "openai": {
    "keys": {
      "openai_key_1": {
        "api_key": "sk-proj-YOUR_ACTUAL_OPENAI_KEY",
        "rate_limits": {
          "gpt-4o": {"rpm": 10, "rpd": 200},
          "gpt-4o-mini": {"rpm": 50, "rpd": 1000}
        }
      }
    }
  }
}
```

**Anthropic Configuration:**
```json
{
  "anthropic": {
    "keys": {
      "anthropic_key_1": {
        "api_key": "sk-ant-api03-YOUR_ACTUAL_ANTHROPIC_KEY",
        "rate_limits": {
          "claude-3-sonnet-20240229": {"rpm": 15, "rpd": 300}
        }
      }
    }
  }
}
```

**Google AI Configuration:**
```json
{
  "gemini": {
    "keys": {
      "gemini_key_1": {
        "api_key": "YOUR_ACTUAL_GEMINI_KEY",
        "rate_limits": {
          "gemini-2.5-flash": {"rpm": 10, "rpd": 250}
        }
      }
    }
  }
}
```

### 4. Rate Limit Configuration

Set appropriate rate limits for your API keys. These limits should be **lower** than your actual API limits to prevent hitting rate limits:

- `rpm`: Requests per minute
- `rpd`: Requests per day

**Example:** If your OpenAI plan allows 50 RPM, set it to 40 RPM to leave a safety margin.

### 5. Model Configuration

Configure models with accurate cost and performance data:

```json
{
  "models": {
    "gpt-4o": {
      "endpoint": "https://api.openai.com/v1/chat/completions",
      "cost_per_1k_tokens": 0.0025,
      "avg_response_time_ms": 4500,
      "intelligence_tier": "high",
      "supports_thinking": false
    }
  }
}
```

**Key Parameters:**
- `cost_per_1k_tokens`: Cost in USD per 1000 tokens
- `avg_response_time_ms`: Average response time in milliseconds
- `intelligence_tier`: "low", "medium", or "high"
- `supports_thinking`: Whether the model supports thinking/reasoning tokens

## Getting API Keys

### OpenAI
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-`)

### Anthropic
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key
5. Copy the key (starts with `sk-ant-`)

### Google AI
1. Go to [ai.google.dev](https://ai.google.dev)
2. Get an API key
3. Copy the key

## First Run

### 1. Start the Server
```bash
python main.py
```

You should see output like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:10006
```

### 2. Test the Health Endpoint
```bash
curl http://localhost:10006/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "mode": "synchronous"
}
```

### 3. Make Your First Request

```bash
curl -X POST "http://localhost:10006/llm/intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "test-client",
    "intelligence_level": "low", 
    "max_wait_seconds": 30,
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

## Environment Variables

You can customize the server using environment variables:

```bash
# Server configuration
HOST=127.0.0.1        # Default: 127.0.0.1
PORT=10006             # Default: 10006
LOG_LEVEL=info        # Default: info (options: debug, info, warning, error)

# Configuration paths
CONFIG_DIR=./config   # Default: ./config
RESULTS_DIR=./results # Default: ./results

# Example usage
HOST=0.0.0.0 PORT=8080 LOG_LEVEL=debug python main.py
```

## Verification Steps

### 1. Check Provider Status
```bash
curl http://localhost:10006/providers
```

This will show the status of all configured providers and models.

### 2. Test Intelligence Routing
```bash
# Low intelligence request (should route to cheap models)
curl -X POST "http://localhost:10006/llm/intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "test",
    "intelligence_level": "low",
    "max_wait_seconds": 30,
    "messages": [{"role": "user", "content": "Hello"}],
    "debug": true
  }'
```

### 3. Check Rate Limits
```bash
curl http://localhost:10006/rates/status
```

## Common Issues

### Issue: "No models available for intelligence level"
**Solution:** Check that you have models configured with the appropriate `intelligence_tier` in `providers.json`.

### Issue: API key errors
**Solution:** 
1. Verify API keys are correct and active
2. Check that the keys have the necessary permissions
3. Ensure rate limits are not exceeded

### Issue: "Provider not found"
**Solution:** Make sure the provider is properly configured in `providers.json` with both `keys` and `models` sections.

### Issue: High response times
**Solution:**
1. Adjust `avg_response_time_ms` values to match actual performance
2. Check network connectivity to API providers
3. Consider increasing `max_wait_seconds` in requests

## Security Considerations

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive configuration in production
3. **Rotate API keys** regularly
4. **Monitor usage** to detect unexpected patterns
5. **Set appropriate rate limits** to prevent abuse

## Next Steps

- Read the [API Documentation](API.md)
- Explore [Advanced Configuration](CONFIGURATION.md)
- Check out [Usage Examples](EXAMPLES.md)
- Learn about [Deployment Options](DEPLOYMENT.md)
