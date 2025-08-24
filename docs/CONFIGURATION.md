# Configuration Guide

Comprehensive guide to configuring the bkvy for optimal performance.

## Configuration Files

The router uses two main configuration files:

- `config/providers.json`: API keys, models, and rate limits
- `config/routing.json`: Scenario-based routing definitions

## Providers Configuration (`providers.json`)

### File Structure

```json
{
  "provider_name": {
    "keys": {
      "key_id": {
        "api_key": "actual_api_key",
        "rate_limits": {
          "model_name": {"rpm": 10, "rpd": 200}
        }
      }
    },
    "models": {
      "model_name": {
        "endpoint": "https://api.provider.com/endpoint",
        "cost_per_1k_tokens": 0.002,
        "avg_response_time_ms": 2000,
        "intelligence_tier": "medium",
        "supports_thinking": true,
        "version": "optional_version"
      }
    }
  }
}
```

### Provider Configurations

#### OpenAI Configuration

```json
{
  "openai": {
    "keys": {
      "openai_key_1": {
        "api_key": "sk-proj-your-actual-key",
        "rate_limits": {
          "gpt-4o": {"rpm": 10, "rpd": 200},
          "gpt-4o-mini": {"rpm": 50, "rpd": 1000},
          "o1-preview": {"rpm": 5, "rpd": 50}
        }
      }
    },
    "models": {
      "gpt-4o": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "cost_per_1k_tokens": 0.0025,
        "avg_response_time_ms": 4500,
        "intelligence_tier": "high",
        "supports_thinking": false
      },
      "gpt-4o-mini": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "cost_per_1k_tokens": 0.00015,
        "avg_response_time_ms": 2000,
        "intelligence_tier": "medium",
        "supports_thinking": false
      },
      "o1-preview": {
        "endpoint": "https://api.openai.com/v1/responses",
        "cost_per_1k_tokens": 0.015,
        "avg_response_time_ms": 10006,
        "intelligence_tier": "high",
        "supports_thinking": true
      }
    }
  }
}
```

#### Anthropic Configuration

```json
{
  "anthropic": {
    "keys": {
      "anthropic_key_1": {
        "api_key": "sk-ant-api03-your-actual-key",
        "rate_limits": {
          "claude-3-haiku-20240307": {"rpm": 25, "rpd": 500},
          "claude-3-sonnet-20240229": {"rpm": 15, "rpd": 300},
          "claude-3-opus-20240229": {"rpm": 5, "rpd": 100}
        }
      }
    },
    "models": {
      "claude-3-haiku-20240307": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "version": "2023-06-01",
        "cost_per_1k_tokens": 0.00025,
        "avg_response_time_ms": 1800,
        "intelligence_tier": "low",
        "supports_thinking": false
      },
      "claude-3-sonnet-20240229": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "version": "2023-06-01", 
        "cost_per_1k_tokens": 0.003,
        "avg_response_time_ms": 2500,
        "intelligence_tier": "medium",
        "supports_thinking": true
      },
      "claude-3-opus-20240229": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "version": "2023-06-01",
        "cost_per_1k_tokens": 0.015,
        "avg_response_time_ms": 4000,
        "intelligence_tier": "high", 
        "supports_thinking": true
      }
    }
  }
}
```

#### Google AI (Gemini) Configuration

```json
{
  "gemini": {
    "keys": {
      "gemini_key_1": {
        "api_key": "your-actual-gemini-key",
        "rate_limits": {
          "gemini-2.5-flash": {"rpm": 10, "rpd": 250},
          "gemini-2.0-flash": {"rpm": 15, "rpd": 200},
          "gemini-2.5-pro": {"rpm": 3, "rpd": 50}
        }
      }
    },
    "models": {
      "gemini-2.5-flash": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        "cost_per_1k_tokens": 0.0,
        "avg_response_time_ms": 2000,
        "intelligence_tier": "low",
        "supports_thinking": true
      },
      "gemini-2.0-flash": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "cost_per_1k_tokens": 0.0,
        "avg_response_time_ms": 2500,
        "intelligence_tier": "medium",
        "supports_thinking": true
      },
      "gemini-2.5-pro": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        "cost_per_1k_tokens": 0.0,
        "avg_response_time_ms": 3000,
        "intelligence_tier": "high",
        "supports_thinking": true
      }
    }
  }
}
```

### Configuration Parameters

#### API Keys (`keys` section)

- **`api_key`**: The actual API key string
- **`rate_limits`**: Per-model rate limiting configuration
  - `rpm`: Requests per minute (should be below your actual API limit)
  - `rpd`: Requests per day (should be below your actual API quota)

#### Models (`models` section)

- **`endpoint`**: Full API endpoint URL
- **`cost_per_1k_tokens`**: Cost in USD per 1000 tokens (input + output)
- **`avg_response_time_ms`**: Average response time in milliseconds
- **`intelligence_tier`**: Classification for routing (`"low"`, `"medium"`, `"high"`)
- **`supports_thinking`**: Whether model supports reasoning/thinking tokens
- **`version`**: API version (Anthropic only)

### Rate Limit Best Practices

1. **Set Conservative Limits**: Always set limits 10-20% below your actual API limits
2. **Account for Burst Usage**: Consider peak usage patterns
3. **Monitor and Adjust**: Use monitoring endpoints to track actual usage
4. **Multiple Keys**: Distribute load across multiple API keys per provider

**Example Conservative Settings:**
```json
// If your OpenAI plan allows 100 RPM
"gpt-4o": {"rpm": 80, "rpd": 10006}

// If your Anthropic plan allows 50 RPM  
"claude-3-sonnet": {"rpm": 40, "rpd": 4000}
```

## Routing Configuration (`routing.json`)

### File Structure

```json
{
  "scenarios": {
    "scenario_name": [
      {"provider": "provider_name", "model": "model_name", "priority": 1},
      {"provider": "provider_name", "model": "model_name", "priority": 2}
    ]
  }
}
```

### Predefined Scenarios

#### Cost-Optimized Scenarios

```json
{
  "scenarios": {
    "cost_optimized_bulk": [
      {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1},
      {"provider": "openai", "model": "gpt-4o-mini", "priority": 2},
      {"provider": "anthropic", "model": "claude-3-haiku-20240307", "priority": 3}
    ],
    "ultra_cheap": [
      {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1}
    ]
  }
}
```

#### Performance-Optimized Scenarios

```json
{
  "scenarios": {
    "speed_first": [
      {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1},
      {"provider": "openai", "model": "gpt-4o-mini", "priority": 2}
    ],
    "balanced_performance": [
      {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "priority": 1},
      {"provider": "openai", "model": "gpt-4o-mini", "priority": 2},
      {"provider": "gemini", "model": "gemini-2.0-flash", "priority": 3}
    ]
  }
}
```

#### Provider-Specific Scenarios

```json
{
  "scenarios": {
    "anthropic_preferred": [
      {"provider": "anthropic", "model": "claude-3-haiku-20240307", "priority": 1},
      {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "priority": 2},
      {"provider": "anthropic", "model": "claude-3-opus-20240229", "priority": 3}
    ],
    "openai_only": [
      {"provider": "openai", "model": "gpt-4o-mini", "priority": 1},
      {"provider": "openai", "model": "gpt-4o", "priority": 2}
    ],
    "gemini_only_fast": [
      {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 1},
      {"provider": "gemini", "model": "gemini-2.0-flash", "priority": 2}
    ]
  }
}
```

#### Reasoning-Focused Scenarios

```json
{
  "scenarios": {
    "reasoning_models": [
      {"provider": "openai", "model": "o1-preview", "priority": 1},
      {"provider": "anthropic", "model": "claude-3-opus-20240229", "priority": 2},
      {"provider": "gemini", "model": "gemini-2.5-pro", "priority": 3}
    ],
    "complex_analysis": [
      {"provider": "anthropic", "model": "claude-3-opus-20240229", "priority": 1},
      {"provider": "openai", "model": "gpt-4o", "priority": 2}
    ]
  }
}
```

#### Application-Specific Scenarios

```json
{
  "scenarios": {
    "customer_support": [
      {"provider": "anthropic", "model": "claude-3-haiku-20240307", "priority": 1},
      {"provider": "gemini", "model": "gemini-2.5-flash", "priority": 2}
    ],
    "code_generation": [
      {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "priority": 1},
      {"provider": "openai", "model": "gpt-4o", "priority": 2}
    ],
    "content_writing": [
      {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "priority": 1},
      {"provider": "openai", "model": "gpt-4o-mini", "priority": 2}
    ],
    "data_analysis": [
      {"provider": "anthropic", "model": "claude-3-opus-20240229", "priority": 1},
      {"provider": "openai", "model": "gpt-4o", "priority": 2}
    ]
  }
}
```

## Intelligence Tier Configuration

The router automatically maps intelligence levels to models based on their `intelligence_tier` setting:

### Low Intelligence (`"intelligence_tier": "low"`)
- Simple questions and answers
- Basic translations
- Text formatting
- Simple calculations
- **Automatically disables thinking** for cost optimization

**Typical Models:**
- `gemini-2.5-flash`
- `claude-3-haiku-20240307`
- `gpt-4o-mini-2024-07-18`

### Medium Intelligence (`"intelligence_tier": "medium"`)
- Code analysis and generation
- Complex reasoning tasks
- Research summaries
- Business analysis

**Typical Models:**
- `gemini-2.0-flash`
- `claude-3-sonnet-20240229`
- `gpt-4o-mini`

### High Intelligence (`"intelligence_tier": "high"`)
- Advanced research
- Complex problem solving
- Strategic analysis
- Critical reasoning tasks

**Typical Models:**
- `gemini-2.5-pro`
- `claude-3-opus-20240229`
- `gpt-4o`
- `o1-preview`

## Thinking Control Configuration

### Automatic Thinking Control

Models with `"supports_thinking": true` will automatically have thinking disabled for low intelligence requests:

```json
{
  "gemini-2.5-flash": {
    "intelligence_tier": "low",
    "supports_thinking": true
    // Thinking will be auto-disabled for low intelligence requests
  }
}
```

### Provider-Specific Thinking Implementation

**Gemini:**
```json
// Sends: {"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}
```

**OpenAI:**
```json  
// Sends: {"reasoning": {"effort": "low"}}
```

**Anthropic:**
```json
// Sends: {"thinking": {"type": "enabled", "budget_tokens": 0}}
```

## Cost Optimization Configuration

### Accurate Cost Setting

Set realistic costs per 1k tokens:

```json
{
  "models": {
    "gpt-4o": {
      "cost_per_1k_tokens": 0.0025  // $2.50 per 1M tokens
    },
    "claude-3-opus": {
      "cost_per_1k_tokens": 0.015   // $15 per 1M tokens  
    },
    "gemini-2.5-flash": {
      "cost_per_1k_tokens": 0.0     // Free tier
    }
  }
}
```

### Performance Tuning

Set realistic response times for accurate routing:

```json
{
  "models": {
    "gemini-2.5-flash": {
      "avg_response_time_ms": 1500   // Fast model
    },
    "claude-3-opus": {
      "avg_response_time_ms": 5000   // Slower but higher quality
    }
  }
}
```

## Environment-Based Configuration

### Development Configuration

```json
{
  "openai": {
    "keys": {
      "dev_key": {
        "api_key": "sk-dev-key",
        "rate_limits": {
          "gpt-4o-mini": {"rpm": 5, "rpd": 100}  // Conservative limits
        }
      }
    }
  }
}
```

### Production Configuration

```json
{
  "openai": {
    "keys": {
      "prod_key_1": {
        "api_key": "sk-prod-key-1", 
        "rate_limits": {
          "gpt-4o": {"rpm": 50, "rpd": 5000}
        }
      },
      "prod_key_2": {
        "api_key": "sk-prod-key-2",
        "rate_limits": {
          "gpt-4o": {"rpm": 50, "rpd": 5000}  // Distribute load
        }
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **"Provider not found"**
   - Check provider name spelling in requests
   - Ensure provider is configured in `providers.json`

2. **"No models available"**
   - Verify models have correct `intelligence_tier`
   - Check rate limits aren't blocking all models

3. **High costs**
   - Review intelligence tier assignments
   - Enable thinking control for appropriate models
   - Check `cost_per_1k_tokens` accuracy

4. **Slow responses**
   - Adjust `avg_response_time_ms` values
   - Check network connectivity
   - Monitor queue status

### Configuration Testing

```bash
# Test configuration validity
python -c "
import json
with open('config/providers.json') as f:
    config = json.load(f)
print('Configuration is valid JSON')
"

# Test API connectivity
curl -X POST 'http://localhost:10006/llm/intelligence' \
  -H 'Content-Type: application/json' \
  -d '{"client_id":"test","intelligence_level":"low","max_wait_seconds":30,"messages":[{"role":"user","content":"test"}],"debug":true}'
```
