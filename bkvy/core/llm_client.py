"""
LLM client for making API calls to providers
"""

import json
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator

from ..utils.logging import setup_logging

logger = setup_logging()


class LLMClient:
    """Handles actual API calls to LLM providers"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def start(self):
        """Start the HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout for individual requests
        )
    
    async def stop(self):
        """Stop the HTTP session"""
        if self.session:
            await self.session.close()
    
    async def check_ollama_health(self, endpoint: str) -> bool:
        """Check if Ollama server is available and responsive"""
        try:
            # Extract base URL from chat endpoint
            if "/api/chat" in endpoint:
                base_url = endpoint.replace("/api/chat", "")
            else:
                base_url = endpoint
            
            health_endpoint = f"{base_url}/api/version"
            
            logger.info("Checking Ollama health", endpoint=health_endpoint)
            
            async with self.session.get(health_endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("Ollama health check passed", version=result.get("version", "unknown"))
                    return True
                else:
                    logger.warning("Ollama health check failed", status=response.status)
                    return False
        except Exception as e:
            logger.warning("Ollama health check failed with exception", error=str(e))
            return False

    async def _make_api_call(self, provider: str, model: str, api_key: str,
                           messages: List[Dict], options: Dict,
                           endpoint: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Make API call to specific provider"""

        if provider == "gemini":
            return await self._call_gemini(endpoint, api_key, messages, options)
        elif provider == "openai":
            return await self._call_openai(endpoint, api_key, model, messages, options)
        elif provider == "anthropic":
            return await self._call_anthropic(endpoint, api_key, model, messages, options, version)
        elif provider == "ollama":
            return await self._call_ollama(endpoint, api_key, model, messages, options)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _make_api_call_stream(self, provider: str, model: str, api_key: str,
                                    messages: List[Dict], options: Dict,
                                    endpoint: str, version: Optional[str] = None
                                    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Make a STREAMING API call to a provider.

        Yields normalized events:
          {"type": "delta", "content": "<text chunk>"}
          {"type": "done",  "finish_reason": <str>, "usage": {input_tokens, output_tokens, total_tokens}}

        The FIRST event yielded is significant: until a "delta" (or "done") is
        produced, the caller may still fall back to another provider. Any
        exception raised BEFORE the first event is a clean pre-first-token
        failure and is safe to fall back on. An exception AFTER the first delta
        means the stream broke mid-flight (the caller is already committed).
        """
        if provider == "gemini":
            gen = self._stream_gemini(endpoint, api_key, messages, options)
        elif provider == "openai":
            gen = self._stream_openai(endpoint, api_key, model, messages, options)
        elif provider == "anthropic":
            gen = self._stream_anthropic(endpoint, api_key, model, messages, options, version)
        elif provider == "ollama":
            gen = self._stream_ollama(endpoint, api_key, model, messages, options)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        async for event in gen:
            yield event

    @staticmethod
    async def _iter_sse_lines(response) -> AsyncGenerator[str, None]:
        """Yield 'data:' payloads from an SSE response, one per event."""
        async for raw in response.content:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                yield line[len("data:"):].strip()

    async def _stream_gemini(self, endpoint: str, api_key: str,
                             messages: List[Dict], options: Dict
                             ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from Gemini via streamGenerateContent?alt=sse."""
        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append({"parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"parts": [{"text": msg["content"]}], "role": "model"})
            elif msg["role"] == "system":
                contents.insert(0, {"parts": [{"text": f"System: {msg['content']}"}]})

        payload = {"contents": contents}
        if options:
            generation_config = {}
            if options.get("max_tokens") is not None:
                generation_config["maxOutputTokens"] = max(options["max_tokens"], 50)
            if options.get("temperature") is not None:
                generation_config["temperature"] = options["temperature"]
            if options.get("top_p") is not None:
                generation_config["topP"] = options["top_p"]
            if options.get("top_k") is not None:
                generation_config["topK"] = options["top_k"]
            if options.get("disable_thinking"):
                generation_config["thinkingConfig"] = {"thinkingBudget": 0}
            if generation_config:
                payload["generationConfig"] = generation_config

        # Convert ":generateContent" endpoint to streaming form.
        stream_endpoint = endpoint.replace(":generateContent", ":streamGenerateContent")
        sep = "&" if "?" in stream_endpoint else "?"
        stream_endpoint = f"{stream_endpoint}{sep}alt=sse"

        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

        input_tokens = output_tokens = total_tokens = 0
        finish_reason = None
        async with self.session.post(stream_endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Gemini API error {response.status}: {error_text}")
            async for data in self._iter_sse_lines(response):
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                candidates = chunk.get("candidates") or []
                if candidates:
                    cand = candidates[0]
                    fr = cand.get("finishReason")
                    if fr:
                        finish_reason = fr
                    for part in (cand.get("content", {}) or {}).get("parts", []) or []:
                        text = part.get("text")
                        if text:
                            yield {"type": "delta", "content": text}
                usage_md = chunk.get("usageMetadata")
                if usage_md:
                    input_tokens = usage_md.get("promptTokenCount", input_tokens)
                    output_tokens = usage_md.get("candidatesTokenCount", output_tokens)
                    total_tokens = usage_md.get("totalTokenCount", input_tokens + output_tokens)

        yield {
            "type": "done",
            "finish_reason": (finish_reason or "stop"),
            "truncated": finish_reason == "MAX_TOKENS",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens or (input_tokens + output_tokens),
            },
        }

    async def _stream_openai(self, endpoint: str, api_key: str, model: str,
                             messages: List[Dict], options: Dict
                             ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from OpenAI Chat Completions (SSE)."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if options.get("max_tokens") is not None:
            payload["max_tokens"] = options["max_tokens"]
        if options.get("temperature") is not None:
            payload["temperature"] = max(0.0, min(2.0, options["temperature"]))
        if options.get("top_p") is not None:
            payload["top_p"] = max(0.0, min(1.0, options["top_p"]))
        if options.get("stop") is not None:
            payload["stop"] = options["stop"]
        if options.get("disable_thinking"):
            payload["reasoning"] = {"effort": "low"}

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        input_tokens = output_tokens = total_tokens = 0
        finish_reason = None
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error {response.status}: {error_text}")
            async for data in self._iter_sse_lines(response):
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    text = delta.get("content")
                    if text:
                        yield {"type": "delta", "content": text}
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                usage = chunk.get("usage")
                if usage:
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
                    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        yield {
            "type": "done",
            "finish_reason": (finish_reason or "stop"),
            "truncated": finish_reason == "length",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens or (input_tokens + output_tokens),
            },
        }

    async def _stream_anthropic(self, endpoint: str, api_key: str, model: str,
                                messages: List[Dict], options: Dict, version: str
                                ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from Anthropic Messages API (SSE)."""
        anthropic_messages = []
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": options.get("max_tokens", 1024),
            "stream": True,
        }
        if system_message:
            payload["system"] = system_message
        if options.get("temperature") is not None:
            payload["temperature"] = max(0.0, min(1.0, options["temperature"]))
        if options.get("top_p") is not None:
            payload["top_p"] = max(0.0, min(1.0, options["top_p"]))
        if options.get("top_k") is not None:
            payload["top_k"] = max(1, options["top_k"])
        if options.get("stop") is not None:
            payload["stop_sequences"] = options["stop"]
        if options.get("disable_thinking"):
            payload["thinking"] = {"type": "enabled", "budget_tokens": 0}

        headers = {
            "x-api-key": api_key,
            "anthropic-version": version or "2023-06-01",
            "Content-Type": "application/json",
        }

        input_tokens = output_tokens = 0
        finish_reason = None
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Anthropic API error {response.status}: {error_text}")
            async for data in self._iter_sse_lines(response):
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "message_start":
                    usage = (event.get("message", {}) or {}).get("usage", {}) or {}
                    input_tokens = usage.get("input_tokens", input_tokens)
                elif etype == "content_block_delta":
                    delta = event.get("delta", {}) or {}
                    text = delta.get("text")
                    if text:
                        yield {"type": "delta", "content": text}
                elif etype == "message_delta":
                    delta = event.get("delta", {}) or {}
                    if delta.get("stop_reason"):
                        finish_reason = delta["stop_reason"]
                    usage = event.get("usage", {}) or {}
                    if usage.get("output_tokens") is not None:
                        output_tokens = usage["output_tokens"]
                elif etype == "error":
                    err = event.get("error", {}) or {}
                    raise Exception(f"Anthropic stream error: {err.get('message', 'unknown')}")

        yield {
            "type": "done",
            "finish_reason": (finish_reason or "stop"),
            "truncated": finish_reason == "max_tokens",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    async def _stream_ollama(self, endpoint: str, api_key: str, model: str,
                             messages: List[Dict], options: Dict
                             ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from Ollama chat API (newline-delimited JSON)."""
        payload = {"model": model, "messages": messages, "stream": True}
        opt = {}
        if options.get("max_tokens") is not None:
            opt["num_predict"] = options["max_tokens"]
        if options.get("temperature") is not None:
            opt["temperature"] = max(0.0, min(2.0, options["temperature"]))
        if options.get("top_p") is not None:
            opt["top_p"] = max(0.0, min(1.0, options["top_p"]))
        if options.get("top_k") is not None:
            opt["top_k"] = max(1, options["top_k"])
        if options.get("stop") is not None:
            opt["stop"] = options["stop"]
        if opt:
            payload["options"] = opt

        headers = {"Content-Type": "application/json"}

        input_tokens = output_tokens = 0
        finish_reason = None
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Ollama API error {response.status}: {error_text}")
            async for raw in response.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = chunk.get("message") or {}
                text = msg.get("content")
                if text:
                    yield {"type": "delta", "content": text}
                if chunk.get("done"):
                    finish_reason = chunk.get("done_reason", "stop")
                    input_tokens = chunk.get("prompt_eval_count", input_tokens)
                    output_tokens = chunk.get("eval_count", output_tokens)

        yield {
            "type": "done",
            "finish_reason": (finish_reason or "stop"),
            "truncated": False,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
    
    async def _call_gemini(self, endpoint: str, api_key: str, 
                          messages: List[Dict], options: Dict) -> Dict[str, Any]:
        """Call Gemini API with robust response parsing and error handling"""
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append({
                    "parts": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                contents.append({
                    "parts": [{"text": msg["content"]}],
                    "role": "model"
                })
            elif msg["role"] == "system":
                # Add system message as first user message
                contents.insert(0, {
                    "parts": [{"text": f"System: {msg['content']}"}]
                })
        
        payload = {
            "contents": contents
        }
        
        # Add generation config if options provided
        if options:
            generation_config = {}
            if "max_tokens" in options and options["max_tokens"] is not None:
                # Ensure minimum token count for Gemini
                generation_config["maxOutputTokens"] = max(options["max_tokens"], 50)
            if "temperature" in options and options["temperature"] is not None:
                generation_config["temperature"] = options["temperature"]
            if "top_p" in options and options["top_p"] is not None:
                generation_config["topP"] = options["top_p"]
            if "top_k" in options and options["top_k"] is not None:
                generation_config["topK"] = options["top_k"]
            
            # Add thinking control for Gemini
            if "disable_thinking" in options and options["disable_thinking"]:
                generation_config["thinkingConfig"] = {"thinkingBudget": 0}
            
            if generation_config:
                payload["generationConfig"] = generation_config
        
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        logger.info("Making Gemini API call", endpoint=endpoint, payload_size=len(str(payload)), 
                   has_thinking_config="thinkingConfig" in payload.get("generationConfig", {}),
                   thinking_budget=payload.get("generationConfig", {}).get("thinkingConfig", {}).get("thinkingBudget"),
                   disable_thinking_option=options.get("disable_thinking") if options else None)
        
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status == 429:
                error_text = await response.text()
                # Handle rate limiting with specific error for retry logic
                raise Exception(f"Gemini API rate limited {response.status}: {error_text}")
            elif response.status != 200:
                error_text = await response.text()
                raise Exception(f"Gemini API error {response.status}: {error_text}")
            
            result = await response.json()
            logger.info("Gemini API response received", has_candidates=bool(result.get("candidates")))
            
            # Extract content with enhanced error handling
            content = ""
            finish_reason = None
            
            if "candidates" in result and result["candidates"]:
                candidate = result["candidates"][0]
                finish_reason = candidate.get("finishReason")
                
                logger.info("Processing candidate", 
                           candidate_keys=list(candidate.keys()),
                           finish_reason=finish_reason)
                
                # Check for content truncation issues
                if finish_reason == "MAX_TOKENS":
                    logger.warning("Gemini response truncated due to MAX_TOKENS", 
                                 candidate=candidate)
                    # For MAX_TOKENS, try to extract any available partial content
                    # The response structure may be incomplete but may still have some text
                
                # Enhanced content extraction
                if "content" in candidate and isinstance(candidate["content"], dict):
                    content_obj = candidate["content"]
                    
                    # Primary method: content.parts[0].text
                    if "parts" in content_obj and isinstance(content_obj["parts"], list) and content_obj["parts"]:
                        for part in content_obj["parts"]:
                            if isinstance(part, dict) and "text" in part:
                                content = part["text"]
                                logger.info("Extracted content via content.parts.text", 
                                          content_length=len(content))
                                break
                    
                    # Fallback: check if content has direct text
                    if not content and "text" in content_obj:
                        content = content_obj["text"]
                        logger.info("Extracted content via content.text", content_length=len(content))
                
                # Additional fallbacks
                if not content:
                    if "text" in candidate:
                        content = candidate["text"]
                        logger.info("Extracted content via candidate.text", content_length=len(content))
                    elif "message" in candidate:
                        content = str(candidate["message"])
                        logger.info("Extracted content via candidate.message", content_length=len(content))
            
            # Handle empty or problematic responses, but not MAX_TOKENS
            if not content or content.strip() == "":
                if finish_reason == "MAX_TOKENS":
                    # For MAX_TOKENS, return partial content or placeholder
                    logger.warning("Gemini response truncated due to MAX_TOKENS, returning empty response")
                    content = ""  # Return empty content but don't raise exception
                else:
                    error_msg = f"Gemini returned empty content (finish_reason: {finish_reason})"
                    logger.error(error_msg, 
                               result_structure={
                                   "candidates_count": len(result.get("candidates", [])),
                                   "candidate_keys": list(result.get("candidates", [{}])[0].keys()) if result.get("candidates") else [],
                                   "finish_reason": finish_reason
                               })
                    raise Exception(error_msg)
            
            # Get usage metadata based on example structure
            usage_data = result.get("usageMetadata", {})
            input_tokens = usage_data.get("promptTokenCount", 0)
            output_tokens = usage_data.get("candidatesTokenCount", 0)
            total_tokens = usage_data.get("totalTokenCount", input_tokens + output_tokens)
            
            # Fallback token estimation if not provided
            if input_tokens == 0:
                input_tokens = sum(len(msg["content"].split()) for msg in messages)
            if output_tokens == 0 and content:
                output_tokens = len(content.split())
            if total_tokens == 0:
                total_tokens = input_tokens + output_tokens
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                },
                "finish_reason": finish_reason,
                "truncated": finish_reason == "MAX_TOKENS",
                "raw_response": result
            }
    
    async def _call_openai(self, endpoint: str, api_key: str, model: str,
                          messages: List[Dict], options: Dict) -> Dict[str, Any]:
        """Call OpenAI API"""
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Add options with validation
        if "max_tokens" in options and options["max_tokens"] is not None:
            payload["max_tokens"] = options["max_tokens"]
        if "temperature" in options and options["temperature"] is not None:
            payload["temperature"] = max(0.0, min(2.0, options["temperature"]))
        if "top_p" in options and options["top_p"] is not None:
            payload["top_p"] = max(0.0, min(1.0, options["top_p"]))
        if "stop" in options and options["stop"] is not None:
            payload["stop"] = options["stop"]
        
        # Add thinking control for OpenAI (reasoning models)
        if "disable_thinking" in options and options["disable_thinking"]:
            # For reasoning models, use low effort to minimize thinking
            payload["reasoning"] = {"effort": "low"}
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error {response.status}: {error_text}")
            
            result = await response.json()
            
            content = ""
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
            
            usage = result.get("usage", {})
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                "raw_response": result
            }
    
    async def _call_anthropic(self, endpoint: str, api_key: str, model: str,
                             messages: List[Dict], options: Dict, version: str) -> Dict[str, Any]:
        """Call Anthropic API"""
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": options.get("max_tokens", 1024)
        }
        
        if system_message:
            payload["system"] = system_message
        
        # Add options with validation
        if "temperature" in options and options["temperature"] is not None:
            payload["temperature"] = max(0.0, min(1.0, options["temperature"]))
        if "top_p" in options and options["top_p"] is not None:
            payload["top_p"] = max(0.0, min(1.0, options["top_p"]))
        if "top_k" in options and options["top_k"] is not None:
            payload["top_k"] = max(1, options["top_k"])
        if "stop" in options and options["stop"] is not None:
            payload["stop_sequences"] = options["stop"]
        
        # Add thinking control for Anthropic
        if "disable_thinking" in options and options["disable_thinking"]:
            # Disable thinking by setting budget to 0
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": 0
            }
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": version or "2023-06-01",
            "Content-Type": "application/json"
        }
        
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Anthropic API error {response.status}: {error_text}")
            
            result = await response.json()
            
            content = ""
            if "content" in result and result["content"]:
                content = result["content"][0].get("text", "")
            
            usage = result.get("usage", {})
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                },
                "raw_response": result
            }
    
    async def _call_ollama(self, endpoint: str, api_key: str, model: str,
                          messages: List[Dict], options: Dict) -> Dict[str, Any]:
        """Call Ollama API"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        # Add options with validation
        if "max_tokens" in options and options["max_tokens"] is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = options["max_tokens"]
        
        if "temperature" in options and options["temperature"] is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = max(0.0, min(2.0, options["temperature"]))
        
        if "top_p" in options and options["top_p"] is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["top_p"] = max(0.0, min(1.0, options["top_p"]))
        
        if "top_k" in options and options["top_k"] is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["top_k"] = max(1, options["top_k"])
        
        if "stop" in options and options["stop"] is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["stop"] = options["stop"]
        
        # Ollama doesn't require API keys, but we'll include headers for consistency
        headers = {
            "Content-Type": "application/json"
        }
        
        logger.info("Making Ollama API call", endpoint=endpoint, model=model, 
                   payload_size=len(str(payload)))
        
        async with self.session.post(endpoint, json=payload, headers=headers) as response:
            if response.status == 429:
                error_text = await response.text()
                raise Exception(f"Ollama API rate limited {response.status}: {error_text}")
            elif response.status != 200:
                error_text = await response.text()
                raise Exception(f"Ollama API error {response.status}: {error_text}")
            
            result = await response.json()
            logger.info("Ollama API response received", has_message=bool(result.get("message")))
            
            # Extract content from Ollama response format
            content = ""
            finish_reason = "stop"
            
            if "message" in result and isinstance(result["message"], dict):
                message = result["message"]
                if "content" in message:
                    content = message["content"]
                
                # Handle thinking content if present
                if "thinking" in message and message["thinking"]:
                    thinking_content = message["thinking"]
                    logger.info("Ollama returned thinking content", thinking_length=len(thinking_content))
                    # For consistency, we don't include thinking in the main content
                    # but log its presence for debugging
            
            if "done_reason" in result:
                finish_reason = result["done_reason"]
            
            # Handle empty response
            if not content or content.strip() == "":
                error_msg = f"Ollama returned empty content (done_reason: {finish_reason})"
                logger.error(error_msg, result_keys=list(result.keys()))
                raise Exception(error_msg)
            
            # Extract usage/token information from Ollama response
            input_tokens = result.get("prompt_eval_count", 0)
            output_tokens = result.get("eval_count", 0)
            total_tokens = input_tokens + output_tokens
            
            # Fallback token estimation if not provided
            if input_tokens == 0:
                input_tokens = sum(len(msg["content"].split()) for msg in messages)
            if output_tokens == 0 and content:
                output_tokens = len(content.split())
            if total_tokens == 0:
                total_tokens = input_tokens + output_tokens
            
            return {
                "content": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                },
                "finish_reason": finish_reason,
                "raw_response": result
            }