"""
LLM client for making API calls to providers
"""

import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Optional

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