# Usage Examples

Real-world examples of using the bkvy for different scenarios.

## Basic Examples

### Simple Question (Low Intelligence)

**Request:**
```bash
curl -X POST "http://localhost:10006/llm/intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "demo-app",
    "intelligence_level": "low",
    "max_wait_seconds": 30,
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "request_id": "12345678-1234-5678-9012-123456789012",
  "model_used": "gemini-2.5-flash",
  "content": "The capital of France is Paris.",
  "usage": {
    "input_tokens": 8,
    "output_tokens": 7,
    "total_tokens": 15
  },
  "finish_reason": "stop",
  "truncated": false,
  "error": null
}
```

**Why this routing?** Low intelligence requests automatically route to the most cost-effective model (Gemini Flash, which is free).

### Complex Analysis (High Intelligence)

**Request:**
```bash
curl -X POST "http://localhost:10006/llm/intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "research-app",
    "intelligence_level": "high",
    "max_wait_seconds": 60,
    "messages": [
      {
        "role": "user", 
        "content": "Analyze the potential economic impacts of implementing universal basic income in developed countries, considering both macroeconomic effects and behavioral changes."
      }
    ],
    "debug": true
  }'
```

**Response (Debug Mode):**
```json
{
  "success": true,
  "request_id": "87654321-4321-8765-2109-876543210987",
  "provider_used": "anthropic",
  "model_used": "claude-3-opus-20240229",
  "api_key_used": "anthropic_key_1",
  "routing_method": "intelligence",
  "decision_reason": "cheapest_within_estimate_after_1_attempts",
  "response": {
    "content": "Universal Basic Income (UBI) implementation in developed countries would likely produce complex, multifaceted economic impacts...",
    "usage": {
      "input_tokens": 45,
      "output_tokens": 487,
      "total_tokens": 532
    },
    "finish_reason": "stop",
    "truncated": false
  },
  "metadata": {
    "rate_limit_wait_ms": 0,
    "queue_wait_ms": 0,
    "api_response_time_ms": 4200,
    "total_completion_time_ms": 4250,
    "cost_usd": 0.00798,
    "alternatives_considered": [
      {
        "provider": "openai",
        "model": "gpt-4o",
        "estimated_total_time_ms": 4500,
        "cost_usd": 0.00133,
        "reason_not_chosen": "higher_cost_or_longer_time_or_failed_retry"
      }
    ]
  }
}
```

## Application-Specific Scenarios

### Customer Support Chatbot

Use cost-optimized models for customer support with fallbacks:

**Request:**
```bash
curl -X POST "http://localhost:10006/llm/scenario" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "support-bot",
    "scenario": "customer_support",
    "max_wait_seconds": 15,
    "messages": [
      {"role": "system", "content": "You are a helpful customer support assistant."},
      {"role": "user", "content": "I cant log into my account, can you help?"}
    ]
  }'
```

**Routing:** Uses `customer_support` scenario which prioritizes fast, cost-effective models like Claude Haiku or Gemini Flash.

### Code Generation

**Request:**
```bash
curl -X POST "http://localhost:10006/llm/scenario" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "code-assistant",
    "scenario": "code_generation",
    "max_wait_seconds": 45,
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to calculate the Fibonacci sequence up to n terms with memoization for optimization."
      }
    ],
    "options": {
      "max_tokens": 1000,
      "temperature": 0.1
    }
  }'
```

### Content Writing

**Request:**
```bash
curl -X POST "http://localhost:10006/llm/scenario" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "content-writer",
    "scenario": "content_writing", 
    "max_wait_seconds": 40,
    "messages": [
      {
        "role": "user",
        "content": "Write a compelling blog post introduction about the future of renewable energy (200 words)."
      }
    ],
    "options": {
      "max_tokens": 300,
      "temperature": 0.7
    }
  }'
```

## Direct Provider Routing

### Specific Model Requirements

When you need a specific model:

**Request:**
```bash
curl -X POST "http://localhost:10006/llm/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "specific-task",
    "provider": "openai",
    "model": "gpt-4o",
    "max_wait_seconds": 30,
    "messages": [
      {"role": "user", "content": "I specifically need GPT-4o for this task."}
    ]
  }'
```

### Testing Different Providers

**Request:**
```bash
curl -X POST "http://localhost:10006/llm/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "testing",
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "api_key_id": "anthropic_key_1",
    "max_wait_seconds": 30,
    "messages": [
      {"role": "user", "content": "Compare this with other providers."}
    ],
    "debug": true
  }'
```

## Advanced Options

### Conversation Context

**Request:**
```bash
curl -X POST "http://localhost:10006/llm/intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "chat-app",
    "intelligence_level": "medium",
    "max_wait_seconds": 30,
    "messages": [
      {"role": "system", "content": "You are a helpful programming assistant."},
      {"role": "user", "content": "How do I sort a list in Python?"},
      {"role": "assistant", "content": "You can sort a list in Python using the sort() method or sorted() function."},
      {"role": "user", "content": "What about sorting by a custom key?"}
    ]
  }'
```

### Temperature and Creativity Control

**Request:**
```bash
curl -X POST "http://localhost:10006/llm/intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "creative-writer",
    "intelligence_level": "medium",
    "max_wait_seconds": 30,
    "messages": [
      {"role": "user", "content": "Write a creative short story about time travel."}
    ],
    "options": {
      "max_tokens": 500,
      "temperature": 0.9,
      "top_p": 0.95
    }
  }'
```

### Thinking Control

**Disable thinking for simple tasks:**
```bash
curl -X POST "http://localhost:10006/llm/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "simple-task",
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "max_wait_seconds": 30,
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "options": {
      "disable_thinking": true
    }
  }'
```

**Enable thinking for complex reasoning:**
```bash
curl -X POST "http://localhost:10006/llm/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "complex-reasoning",
    "provider": "openai", 
    "model": "o1-preview",
    "max_wait_seconds": 60,
    "messages": [
      {"role": "user", "content": "Solve this complex logic puzzle: ..."}
    ],
    "options": {
      "disable_thinking": false
    }
  }'
```

## Error Handling Examples

### Rate Limit Handling

**Request when rate limited:**
```bash
curl -X POST "http://localhost:10006/llm/intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "rate-limited-client",
    "intelligence_level": "low",
    "max_wait_seconds": 5,
    "messages": [{"role": "user", "content": "Quick question"}]
  }'
```

**Response:**
```json
{
  "success": false,
  "request_id": "rate-limit-example",
  "error_code": "no_combinations_available",
  "message": "No available combinations within time limit",
  "retry_suggestion": {
    "fastest_available_combination": {
      "provider": "gemini",
      "model": "gemini-2.5-flash",
      "api_key": "gemini_key_2",
      "estimated_total_time_seconds": 12.5,
      "available_at": "2024-01-15T10:42:30Z"
    }
  }
}
```

### Provider Failover

**When primary provider fails:**
```bash
# This automatically tries alternatives
curl -X POST "http://localhost:10006/llm/intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "failover-test",
    "intelligence_level": "medium",
    "max_wait_seconds": 30,
    "messages": [{"role": "user", "content": "Test failover"}],
    "debug": true
  }'
```

## Batch Processing Examples

### Multiple Simple Requests

**Shell script for batch processing:**
```bash
#!/bin/bash

# Process multiple simple queries
queries=(
  "What is the weather like?"
  "How do I cook pasta?"
  "What time is it?"
  "Define machine learning"
)

for query in "${queries[@]}"; do
  curl -X POST "http://localhost:10006/llm/intelligence" \
    -H "Content-Type: application/json" \
    -d "{
      \"client_id\": \"batch-processor\",
      \"intelligence_level\": \"low\",
      \"max_wait_seconds\": 30,
      \"messages\": [{\"role\": \"user\", \"content\": \"$query\"}]
    }" &
done

wait  # Wait for all background jobs to complete
```

### Parallel Processing

**Python script for concurrent requests:**
```python
import asyncio
import aiohttp
import json

async def make_request(session, query, client_id):
    data = {
        "client_id": client_id,
        "intelligence_level": "low",
        "max_wait_seconds": 30,
        "messages": [{"role": "user", "content": query}]
    }
    
    async with session.post(
        "http://localhost:10006/llm/intelligence",
        json=data,
        headers={"Content-Type": "application/json"}
    ) as response:
        return await response.json()

async def batch_process():
    queries = [
        "What is Python?",
        "How do I install Node.js?", 
        "What is React?",
        "Explain Docker containers"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            make_request(session, query, f"client-{i}")
            for i, query in enumerate(queries)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        for i, response in enumerate(responses):
            print(f"Query {i}: {response.get('content', 'Error')}")

# Run the batch processor
asyncio.run(batch_process())
```

## Integration Examples

### Express.js Integration

```javascript
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

app.post('/chat', async (req, res) => {
  try {
    const { message, complexity = 'low' } = req.body;
    
    const response = await axios.post('http://localhost:10006/llm/intelligence', {
      client_id: req.ip,
      intelligence_level: complexity,
      max_wait_seconds: 30,
      messages: [{ role: 'user', content: message }]
    });
    
    res.json({
      success: true,
      message: response.data.content,
      model: response.data.model_used
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.listen(3000, () => {
  console.log('Chat API running on port 3000');
});
```

### Python FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from typing import List, Optional

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    complexity: Optional[str] = "medium"
    max_wait: Optional[int] = 30

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:10006/llm/intelligence",
                json={
                    "client_id": "fastapi-integration",
                    "intelligence_level": request.complexity,
                    "max_wait_seconds": request.max_wait,
                    "messages": [msg.dict() for msg in request.messages]
                }
            )
            
            data = response.json()
            
            if data["success"]:
                return {
                    "message": data["content"],
                    "model": data["model_used"]
                }
            else:
                raise HTTPException(status_code=500, detail=data["message"])
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
```

### React Frontend Integration

```javascript
import React, { useState } from 'react';
import axios from 'axios';

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:10006/llm/intelligence', {
        client_id: 'react-chat',
        intelligence_level: 'medium',
        max_wait_seconds: 30,
        messages: [...messages, userMessage]
      });

      if (response.data.success) {
        const aiMessage = { 
          role: 'assistant', 
          content: response.data.content,
          model: response.data.model_used
        };
        setMessages(prev => [...prev, aiMessage]);
      }
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <strong>{msg.role}:</strong> {msg.content}
            {msg.model && <small> (via {msg.model})</small>}
          </div>
        ))}
      </div>
      
      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          disabled={loading}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default ChatInterface;
```

## Monitoring and Analytics

### Cost Tracking

```bash
# Get detailed cost information
curl -X POST "http://localhost:10006/llm/intelligence" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "cost-tracker",
    "intelligence_level": "high", 
    "max_wait_seconds": 30,
    "messages": [{"role": "user", "content": "Expensive request"}],
    "debug": true
  }' | jq '.metadata.cost_usd'
```

### Performance Monitoring

```bash
# Monitor response times
curl -s "http://localhost:10006/providers" | \
  jq '.providers[] | .models[] | {model: .model, avg_time: .avg_response_time_ms}'
```

### Usage Analytics

```python
import requests
import json
from datetime import datetime

def track_usage(client_id, intelligence_level, cost, tokens):
    """Track usage for analytics"""
    usage_data = {
        'timestamp': datetime.now().isoformat(),
        'client_id': client_id,
        'intelligence_level': intelligence_level,
        'cost_usd': cost,
        'tokens_used': tokens
    }
    
    # Log to file or database
    with open('usage_log.jsonl', 'a') as f:
        f.write(json.dumps(usage_data) + '\\n')

# Use after API calls
response = requests.post('http://localhost:10006/llm/intelligence', json={
    'client_id': 'analytics-test',
    'intelligence_level': 'medium',
    'max_wait_seconds': 30,
    'messages': [{'role': 'user', 'content': 'Test message'}],
    'debug': True
})

if response.json()['success']:
    data = response.json()
    track_usage(
        client_id='analytics-test',
        intelligence_level='medium',
        cost=data['metadata']['cost_usd'],
        tokens=data['response']['usage']['total_tokens']
    )
```

## Testing and Development

### Load Testing

```bash
# Simple load test
ab -n 100 -c 10 -p test_payload.json -T application/json \
  http://localhost:10006/llm/intelligence
```

### A/B Testing Different Scenarios

```python
import random
import requests

scenarios = ['cost_optimized', 'balanced_performance', 'speed_first']

def ab_test_scenarios(query):
    scenario = random.choice(scenarios)
    
    response = requests.post('http://localhost:10006/llm/scenario', json={
        'client_id': f'ab-test-{scenario}',
        'scenario': scenario,
        'max_wait_seconds': 30,
        'messages': [{'role': 'user', 'content': query}],
        'debug': True
    })
    
    if response.json()['success']:
        data = response.json()
        return {
            'scenario': scenario,
            'model': data['model_used'],
            'cost': data['metadata']['cost_usd'],
            'time_ms': data['metadata']['total_completion_time_ms']
        }

# Test with multiple queries
results = [ab_test_scenarios("What is machine learning?") for _ in range(10)]
print(json.dumps(results, indent=2))
```

These examples demonstrate the flexibility and power of the bkvy for various use cases, from simple questions to complex integrations and monitoring setups.
