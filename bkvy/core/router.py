"""
Intelligent router for LLM requests
"""

import asyncio
import time
import uuid
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

from ..models.enums import RoutingMethod
from ..models.data_classes import CompletionTimeAnalysis
from ..models.schemas import (
    IntelligenceRequest, ScenarioRequest, DirectRequest, 
    LLMResponse, SimplifiedResponse, ResponseMetadata, Message, LLMOptions
)
from ..utils.logging import setup_logging

logger = setup_logging()


class IntelligentRouter:
    """Core routing engine with intelligent combination selection - NOW SYNCHRONOUS"""
    
    def __init__(self, config_manager, rate_limit_manager, queue_manager, llm_client):
        self.config = config_manager
        self.rate_limits = rate_limit_manager
        self.queues = queue_manager
        self.llm_client = llm_client
    
    def _create_simplified_response(self, full_response: LLMResponse) -> SimplifiedResponse:
        """Convert a full response to a simplified response"""
        if full_response.success:
            response_data = full_response.response or {}
            return SimplifiedResponse(
                success=True,
                request_id=full_response.request_id,
                model_used=full_response.model_used,
                content=response_data.get("content"),
                usage=response_data.get("usage"),
                finish_reason=response_data.get("finish_reason"),
                truncated=response_data.get("truncated"),
                error=None
            )
        else:
            return SimplifiedResponse(
                success=False,
                request_id=full_response.request_id,
                model_used=full_response.model_used,
                content=None,
                usage=None,
                finish_reason=None,
                truncated=None,
                error=full_response.message or full_response.error_code
            )
    
    async def route_intelligence_request(self, request: IntelligenceRequest) -> LLMResponse:
        """Route request based on intelligence level - WAITS FOR COMPLETION WITH RETRY LOGIC"""
        logger.info("Processing intelligence-based request", 
                   client_id=request.client_id,
                   intelligence_level=request.intelligence_level,
                   max_wait=request.max_wait_seconds)
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Get all models matching intelligence level
        model_combinations = self.config.get_models_by_intelligence(request.intelligence_level.value)
        
        logger.info("🔍 INTELLIGENCE DEBUG: Found models for intelligence level", 
                   intelligence_level=request.intelligence_level.value,
                   combinations=model_combinations,
                   count=len(model_combinations))
        
        if not model_combinations:
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="no_models_available",
                message=f"No models available for intelligence level: {request.intelligence_level.value}"
            )
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
        
        # Analyze all combinations (use max_wait_seconds for route calculation only)
        logger.info("🔍 ANALYSIS DEBUG: Starting analysis of all combinations")
        analyses = await self._analyze_all_combinations(model_combinations, request.max_wait_seconds)
        
        logger.info("🔍 ANALYSIS DEBUG: Analysis complete", 
                   valid_analyses=len(analyses),
                   total_combinations=len(model_combinations))
        
        if not analyses:
            return await self._create_failure_response(model_combinations, request.max_wait_seconds, "intelligence", request_id)
        
        # Sort by best combination (cheapest, then fastest)
        sorted_analyses = sorted(analyses, key=lambda x: (x.cost_per_1k_tokens, x.total_seconds))
        
        logger.info("🔍 SORTED DEBUG: All alternatives in order", 
                   alternatives_count=len(sorted_analyses),
                   alternatives=[{
                       "rank": i+1,
                       "provider": analysis.provider,
                       "model": analysis.model,
                       "api_key_id": analysis.api_key_id,
                       "cost": analysis.cost_per_1k_tokens,
                       "total_seconds": analysis.total_seconds
                   } for i, analysis in enumerate(sorted_analyses)])
        
        logger.info("🚀 RETRY DEBUG: Attempting request with retry logic", 
                   alternatives_count=len(sorted_analyses))
        
        # Try each alternative with retry logic - WITH EXCEPTION SAFETY
        try:
            result, used_analysis, attempt_info = await self._execute_with_retry_and_fallback(
                sorted_analyses, request.messages, request.options
            )
        except Exception as e:
            logger.error("💥 FATAL ERROR in retry logic", 
                        error=str(e),
                        traceback=traceback.format_exc())
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="retry_logic_failure",
                message=f"Fatal error in retry logic: {str(e)}"
            )
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
        
        total_time = time.time() - start_time
        
        if result["success"]:
            full_response = LLMResponse(
                success=True,
                request_id=request_id,
                provider_used=used_analysis.provider,
                model_used=used_analysis.model,
                api_key_used=used_analysis.api_key_id,
                routing_method=RoutingMethod.INTELLIGENCE,
                decision_reason=f"cheapest_within_estimate_after_{attempt_info['total_attempts']}_attempts",
                response=result["response"],
                metadata=self._create_metadata_with_attempts(used_analysis, sorted_analyses, total_time, attempt_info)
            )
            
            # Return simplified response if debug is False
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
        else:
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="all_alternatives_failed",
                message=f"All {len(sorted_analyses)} alternatives failed after retry attempts. Last error: {result.get('error', 'Unknown error')}",
                evaluated_combinations=self._create_failure_summary(sorted_analyses, attempt_info)
            )
            
            # Return simplified response if debug is False
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
    
    async def route_scenario_request(self, request: ScenarioRequest) -> LLMResponse:
        """Route request based on scenario - WAITS FOR COMPLETION WITH RETRY LOGIC"""
        logger.info("Processing scenario-based request",
                   client_id=request.client_id,
                   scenario=request.scenario,
                   max_wait=request.max_wait_seconds)
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Get scenario combinations
        scenario_combinations = self.config.get_scenario_combinations(request.scenario)
        
        if not scenario_combinations:
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="scenario_not_found",
                message=f"Scenario not found: {request.scenario}"
            )
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
        
        # Convert to (provider, model) tuples and analyze
        model_combinations = [(provider, model) for provider, model, _ in scenario_combinations]
        analyses = await self._analyze_all_combinations(model_combinations, request.max_wait_seconds)
        
        if not analyses:
            return await self._create_failure_response(model_combinations, request.max_wait_seconds, "scenario", request_id)
        
        # Apply scenario priorities and sort
        priority_map = {(provider, model): priority for provider, model, priority in scenario_combinations}
        
        # Sort by priority first, then cost, then time
        sorted_analyses = sorted(analyses, key=lambda x: (
            priority_map.get((x.provider, x.model), 999),
            x.cost_per_1k_tokens,
            x.total_seconds
        ))
        
        logger.info("Attempting scenario request with retry logic", 
                   scenario=request.scenario,
                   alternatives_count=len(sorted_analyses))
        
        # Try each alternative with retry logic
        result, used_analysis, attempt_info = await self._execute_with_retry_and_fallback(
            sorted_analyses, request.messages, request.options
        )
        
        total_time = time.time() - start_time
        
        if result["success"]:
            full_response = LLMResponse(
                success=True,
                request_id=request_id,
                provider_used=used_analysis.provider,
                model_used=used_analysis.model,
                api_key_used=used_analysis.api_key_id,
                routing_method=RoutingMethod.SCENARIO,
                decision_reason=f"scenario_priority_cost_optimized_after_{attempt_info['total_attempts']}_attempts",
                response=result["response"],
                metadata=self._create_metadata_with_attempts(used_analysis, analyses, total_time, attempt_info)
            )
            
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
        else:
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="all_alternatives_failed",
                message=f"All {len(sorted_analyses)} scenario alternatives failed after retry attempts. Last error: {result.get('error', 'Unknown error')}",
                evaluated_combinations=self._create_failure_summary(sorted_analyses, attempt_info)
            )
            
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
    
    async def route_direct_request(self, request: DirectRequest) -> LLMResponse:
        """Route request to specific provider/model - WAITS FOR COMPLETION WITH RETRY LOGIC"""
        logger.info("Processing direct request",
                   client_id=request.client_id,
                   provider=request.provider,
                   model=request.model_name,
                   api_key_id=request.api_key_id)
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Validate provider and model
        if request.provider not in self.config.providers:
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="provider_not_found",
                message=f"Provider not found: {request.provider}"
            )
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
        
        provider_config = self.config.providers[request.provider]
        
        if request.model_name not in provider_config.models:
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="model_not_found",
                message=f"Model not found: {request.model_name}"
            )
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
        
        # Determine API key(s) to use
        api_keys_to_check = []
        if request.api_key_id:
            if request.api_key_id in provider_config.keys:
                api_keys_to_check = [request.api_key_id]
            else:
                full_response = LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="api_key_not_found",
                    message=f"API key not found: {request.api_key_id}"
                )
                if not request.debug:
                    return self._create_simplified_response(full_response)
                return full_response
        else:
            # Auto-select from all available keys
            api_keys_to_check = list(provider_config.keys.keys())
        
        # Analyze combinations
        model_combinations = [(request.provider, request.model_name)]
        analyses = await self._analyze_combinations_with_keys(
            model_combinations, api_keys_to_check, request.max_wait_seconds
        )
        
        if not analyses:
            return await self._create_failure_response(
                model_combinations, request.max_wait_seconds, "direct", request_id
            )
        
        # Sort by fastest completion time for direct requests
        sorted_analyses = sorted(analyses, key=lambda x: x.total_seconds)
        
        logger.info("Attempting direct request with retry logic", 
                   provider=request.provider,
                   model=request.model_name,
                   alternatives_count=len(sorted_analyses))
        
        # Try each alternative with retry logic
        result, used_analysis, attempt_info = await self._execute_with_retry_and_fallback(
            sorted_analyses, request.messages, request.options
        )
        
        total_time = time.time() - start_time
        
        if result["success"]:
            full_response = LLMResponse(
                success=True,
                request_id=request_id,
                provider_used=used_analysis.provider,
                model_used=used_analysis.model,
                api_key_used=used_analysis.api_key_id,
                routing_method=RoutingMethod.DIRECT,
                decision_reason=f"direct_fastest_completion_after_{attempt_info['total_attempts']}_attempts",
                response=result["response"],
                metadata=self._create_metadata_with_attempts(used_analysis, analyses, total_time, attempt_info)
            )
            
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
        else:
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="all_alternatives_failed",
                message=f"All {len(sorted_analyses)} direct alternatives failed after retry attempts. Last error: {result.get('error', 'Unknown error')}",
                evaluated_combinations=self._create_failure_summary(sorted_analyses, attempt_info)
            )
            
            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
    
    async def _execute_with_retry_and_fallback(self, sorted_analyses: List[CompletionTimeAnalysis],
                                             messages: List[Message], options: Optional[LLMOptions]) -> Tuple[Dict[str, Any], CompletionTimeAnalysis, Dict[str, Any]]:
        """Execute request with retry logic and automatic failover - enhanced error handling"""
        MAX_RETRIES = 3
        attempt_info = {
            "alternatives_tried": 0,
            "total_attempts": 0,
            "failures": []
        }
        
        for analysis in sorted_analyses:
            attempt_info["alternatives_tried"] += 1
            
            logger.info("Trying alternative", 
                       alternative_num=attempt_info["alternatives_tried"],
                       provider=analysis.provider,
                       model=analysis.model,
                       api_key_id=analysis.api_key_id)
            
            # Try this alternative up to MAX_RETRIES times
            for retry_attempt in range(1, MAX_RETRIES + 1):
                attempt_info["total_attempts"] += 1
                
                logger.info("Attempt", 
                           alternative=attempt_info["alternatives_tried"],
                           retry=retry_attempt,
                           total_attempts=attempt_info["total_attempts"],
                           provider=analysis.provider,
                           model=analysis.model)
                
                try:
                    result = await self._execute_request(analysis, messages, options)
                    
                    if result["success"]:
                        logger.info("SUCCESS", 
                                   alternative=attempt_info["alternatives_tried"],
                                   retry=retry_attempt,
                                   provider=analysis.provider,
                                   model=analysis.model)
                        return result, analysis, attempt_info
                    else:
                        # Check if this is a failure that should skip retries and move to next provider
                        error_msg = result.get("error", "Unknown error")
                        should_skip_retries = self._should_skip_retries(error_msg)
                        
                        logger.warning("Attempt failed", 
                                     alternative=attempt_info["alternatives_tried"],
                                     retry=retry_attempt,
                                     provider=analysis.provider,
                                     model=analysis.model,
                                     error=error_msg,
                                     skip_retries=should_skip_retries)
                        
                        attempt_info["failures"].append({
                            "alternative": attempt_info["alternatives_tried"],
                            "retry": retry_attempt,
                            "provider": analysis.provider,
                            "model": analysis.model,
                            "api_key_id": analysis.api_key_id,
                            "error": error_msg,
                            "skip_retries": should_skip_retries
                        })
                        
                        # If should skip retries or this was the last retry, break to try next alternative
                        if should_skip_retries or retry_attempt == MAX_RETRIES:
                            if should_skip_retries:
                                logger.info("Skipping retries due to error type, moving to next provider", 
                                           provider=analysis.provider,
                                           model=analysis.model,
                                           error=error_msg)
                            else:
                                logger.error("Alternative exhausted after retries", 
                                           provider=analysis.provider,
                                           model=analysis.model,
                                           api_key_id=analysis.api_key_id,
                                           retries=MAX_RETRIES)
                            break
                        else:
                            # Wait a bit before retry (exponential backoff)
                            wait_time = min(2 ** (retry_attempt - 1), 5)  # 1s, 2s, 4s max
                            logger.info("Retrying after wait", 
                                       wait_seconds=wait_time,
                                       next_retry=retry_attempt + 1)
                            await asyncio.sleep(wait_time)
                            
                except Exception as e:
                    # Unexpected error (not from API response)
                    error_msg = f"Unexpected error: {str(e)}"
                    should_skip_retries = self._should_skip_retries(str(e))
                    
                    logger.error("Unexpected error during attempt", 
                               alternative=attempt_info["alternatives_tried"],
                               retry=retry_attempt,
                               provider=analysis.provider,
                               model=analysis.model,
                               error=error_msg,
                               skip_retries=should_skip_retries,
                               traceback=traceback.format_exc())
                    
                    attempt_info["failures"].append({
                        "alternative": attempt_info["alternatives_tried"],
                        "retry": retry_attempt,
                        "provider": analysis.provider,
                        "model": analysis.model,
                        "api_key_id": analysis.api_key_id,
                        "error": error_msg,
                        "skip_retries": should_skip_retries
                    })
                    
                    # For rate limiting or critical errors, skip retries
                    if should_skip_retries or retry_attempt == MAX_RETRIES:
                        break
                    else:
                        wait_time = min(2 ** (retry_attempt - 1), 5)
                        await asyncio.sleep(wait_time)
        
        # All alternatives exhausted
        last_error = attempt_info["failures"][-1]["error"] if attempt_info["failures"] else "No alternatives available"
        logger.error("All alternatives exhausted", 
                   alternatives_tried=attempt_info["alternatives_tried"],
                   total_attempts=attempt_info["total_attempts"],
                   last_error=last_error)
        
        return {
            "success": False,
            "error": f"All {attempt_info['alternatives_tried']} alternatives failed after {attempt_info['total_attempts']} total attempts"
        }, sorted_analyses[0] if sorted_analyses else None, attempt_info
    
    def _should_skip_retries(self, error_msg: str) -> bool:
        """Determine if an error should skip retries and move to next provider"""
        error_lower = error_msg.lower()
        
        # Rate limiting errors - skip retries
        if any(phrase in error_lower for phrase in [
            "rate limited", "429", "quota", "exceeded", "resource_exhausted"
        ]):
            return True
            
        # Authentication errors - skip retries  
        if any(phrase in error_lower for phrase in [
            "401", "403", "unauthorized", "forbidden", "invalid api key"
        ]):
            return True
            
        # Content/response errors that won't be fixed by retrying
        if any(phrase in error_lower for phrase in [
            "empty content", "could not extract content", "max_tokens"
        ]):
            return True
            
        # Network errors that might be temporary - allow retries
        if any(phrase in error_lower for phrase in [
            "timeout", "connection", "network", "500", "502", "503", "504"
        ]):
            return False
            
        # Default: allow retries for unknown errors
        return False
    
    async def _execute_request(self, analysis: CompletionTimeAnalysis, 
                             messages: List[Message], options: Optional[LLMOptions]) -> Dict[str, Any]:
        """Execute request to selected combination and wait for response - NO TIMEOUT"""
        provider_config = self.config.providers[analysis.provider]
        key_config = provider_config.keys[analysis.api_key_id]
        model_config = provider_config.models[analysis.model]
        
        # Prepare options with automatic thinking control
        options_dict = options.dict() if options else {}
        
        # Automatically disable thinking for low intelligence models if not explicitly set
        if options_dict.get("disable_thinking") is None:
            if model_config.intelligence_tier == "low" and model_config.supports_thinking:
                options_dict["disable_thinking"] = True
                logger.info("Auto-disabled thinking for low intelligence model", 
                           provider=analysis.provider, model=analysis.model, 
                           intelligence_tier=model_config.intelligence_tier,
                           supports_thinking=model_config.supports_thinking)
        
        request_data = {
            "provider": analysis.provider,
            "model": analysis.model,
            "api_key": key_config.api_key,
            "endpoint": model_config.endpoint,
            "version": getattr(model_config, 'version', None),
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "options": options_dict
        }
        
        # Execute directly and wait for response (no timeout - requests complete regardless of time)
        result = await self.queues.execute_request_directly(
            analysis.provider, analysis.model, analysis.api_key_id, request_data, 0,  # max_wait_seconds=0 (ignored)
            self.rate_limits, self.config, self.llm_client
        )
        
        return result
    
    def _create_metadata_with_attempts(self, selected: CompletionTimeAnalysis, 
                                     all_analyses: List[CompletionTimeAnalysis], 
                                     actual_time: float, attempt_info: Dict[str, Any]) -> ResponseMetadata:
        """Create response metadata including retry attempt information"""
        alternatives = []
        for analysis in all_analyses:
            if analysis.combination_key != selected.combination_key:
                alternatives.append({
                    "provider": analysis.provider,
                    "model": analysis.model,
                    "api_key": analysis.api_key_id,
                    "estimated_total_time_ms": int(analysis.total_seconds * 1000),
                    "cost_usd": analysis.cost_per_1k_tokens,
                    "reason_not_chosen": "higher_cost_or_longer_time_or_failed_retry"
                })
        
        return ResponseMetadata(
            rate_limit_wait_ms=int(selected.rate_limit_wait_seconds * 1000),
            queue_wait_ms=int(selected.queue_wait_seconds * 1000),
            api_response_time_ms=int(selected.processing_time_seconds * 1000),
            total_completion_time_ms=int(actual_time * 1000),
            cost_usd=selected.cost_per_1k_tokens,
            alternatives_considered=alternatives
        )
    
    def _create_failure_summary(self, sorted_analyses: List[CompletionTimeAnalysis], 
                              attempt_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed failure summary for debugging"""
        summary = []
        
        for i, analysis in enumerate(sorted_analyses, 1):
            # Find failures for this alternative
            alternative_failures = [f for f in attempt_info["failures"] if f["alternative"] == i]
            
            summary.append({
                "alternative_rank": i,
                "provider": analysis.provider,
                "model": analysis.model,
                "api_key": analysis.api_key_id,
                "estimated_total_time_seconds": analysis.total_seconds,
                "cost_per_1k_tokens": analysis.cost_per_1k_tokens,
                "retry_attempts": len(alternative_failures),
                "failures": alternative_failures
            })
        
        return summary
    
    async def _analyze_all_combinations(self, model_combinations: List[Tuple[str, str]], 
                                      max_wait_seconds: int) -> List[CompletionTimeAnalysis]:
        """Analyze all (provider, model, api_key) combinations for routing calculations"""
        analyses = []
        
        logger.info("🔍 ANALYSIS DEBUG: Starting analysis of combinations", 
                   total_combinations=len(model_combinations),
                   combinations=model_combinations,
                   max_wait_seconds=max_wait_seconds)
        
        for provider, model in model_combinations:
            logger.info("🔍 PROVIDER DEBUG: Analyzing provider/model", 
                       provider=provider, model=model)
            
            if provider not in self.config.providers:
                logger.warning("❌ PROVIDER DEBUG: Provider not found in config", 
                             provider=provider,
                             available_providers=list(self.config.providers.keys()))
                continue
            
            provider_config = self.config.providers[provider]
            
            if model not in provider_config.models:
                logger.warning("❌ MODEL DEBUG: Model not found in provider config", 
                             provider=provider, model=model,
                             available_models=list(provider_config.models.keys()))
                continue
            
            logger.info("✅ PROVIDER DEBUG: Provider and model found, checking API keys",
                       provider=provider, model=model,
                       available_keys=list(provider_config.keys.keys()))
            
            # Check all API keys for this provider/model combination
            key_count = 0
            for api_key_id, key_config in provider_config.keys.items():
                key_count += 1
                logger.info("🔑 KEY DEBUG: Checking API key", 
                           provider=provider, model=model, api_key_id=api_key_id,
                           key_num=f"{key_count}/{len(provider_config.keys)}")
                
                # Check if this key supports this model
                if model not in key_config.rate_limits:
                    logger.warning("❌ KEY DEBUG: Model not in rate limits for key", 
                                 provider=provider, model=model, api_key_id=api_key_id,
                                 available_models=list(key_config.rate_limits.keys()))
                    continue
                
                logger.info("✅ KEY DEBUG: Key supports model, performing analysis",
                           provider=provider, model=model, api_key_id=api_key_id)
                
                analysis = await self._analyze_combination(
                    provider, model, api_key_id, max_wait_seconds
                )
                
                if analysis:
                    logger.info("✅ ANALYSIS DEBUG: Analysis completed successfully", 
                               provider=provider, model=model, api_key_id=api_key_id,
                               total_seconds=analysis.total_seconds,
                               cost=analysis.cost_per_1k_tokens,
                               within_estimate=analysis.total_seconds <= max_wait_seconds)
                    
                    # Include all analyses (remove time filter for routing)
                    analyses.append(analysis)
                    logger.info("✅ ADDED DEBUG: Analysis added to alternatives",
                               provider=provider, model=model, api_key_id=api_key_id,
                               total_valid_alternatives=len(analyses))
                else:
                    logger.error("❌ ANALYSIS DEBUG: Analysis failed", 
                               provider=provider, model=model, api_key_id=api_key_id)
        
        logger.info("🏁 ANALYSIS DEBUG: Analysis complete", 
                   total_combinations_checked=len(model_combinations),
                   valid_combinations=len(analyses),
                   alternatives=[{
                       "provider": a.provider,
                       "model": a.model, 
                       "api_key_id": a.api_key_id,
                       "cost": a.cost_per_1k_tokens,
                       "total_seconds": a.total_seconds
                   } for a in analyses])
        
        return analyses
    
    async def _analyze_combinations_with_keys(self, model_combinations: List[Tuple[str, str]],
                                            api_key_ids: List[str], 
                                            max_wait_seconds: int) -> List[CompletionTimeAnalysis]:
        """Analyze specific (provider, model, api_key) combinations"""
        analyses = []
        
        for provider, model in model_combinations:
            if provider not in self.config.providers:
                continue
            
            provider_config = self.config.providers[provider]
            
            if model not in provider_config.models:
                continue
            
            for api_key_id in api_key_ids:
                if api_key_id not in provider_config.keys:
                    continue
                
                key_config = provider_config.keys[api_key_id]
                
                # Check if this key supports this model
                if model not in key_config.rate_limits:
                    continue
                
                analysis = await self._analyze_combination(
                    provider, model, api_key_id, max_wait_seconds
                )
                
                if analysis:
                    analyses.append(analysis)
        
        return analyses
    
    async def _analyze_combination(self, provider: str, model: str, api_key_id: str,
                                 max_wait_seconds: int) -> Optional[CompletionTimeAnalysis]:
        """Analyze a specific (provider, model, api_key) combination"""
        try:
            provider_config = self.config.providers[provider]
            key_config = provider_config.keys[api_key_id]
            model_config = provider_config.models[model]
            
            # Get rate limits for this combination
            rate_limits = key_config.rate_limits[model]
            rpm_limit = rate_limits["rpm"]
            rpd_limit = rate_limits["rpd"]
            
            # Check rate limit status
            is_limited, rate_wait = await self.rate_limits.check_rate_limit_status(
                provider, model, api_key_id, rpm_limit, rpd_limit
            )
            
            # Get queue wait time
            queue_wait = await self.queues.get_queue_wait_time(
                provider, model, api_key_id, model_config.avg_response_time_ms
            )
            
            # Calculate total completion time
            processing_time = model_config.avg_response_time_ms / 1000  # Convert to seconds
            total_time = rate_wait + queue_wait + processing_time
            
            combination_key = f"{provider}_{api_key_id}_{model}"
            
            return CompletionTimeAnalysis(
                rate_limit_wait_seconds=rate_wait,
                queue_wait_seconds=queue_wait,
                processing_time_seconds=processing_time,
                total_seconds=total_time,
                cost_per_1k_tokens=model_config.cost_per_1k_tokens,
                combination_key=combination_key,
                provider=provider,
                model=model,
                api_key_id=api_key_id
            )
            
        except Exception as e:
            logger.error("Error analyzing combination", 
                        provider=provider, model=model, api_key_id=api_key_id, error=str(e))
            return None
    
    async def _create_failure_response(self, model_combinations: List[Tuple[str, str]],
                                     max_wait_seconds: int, routing_method: str, request_id: str) -> LLMResponse:
        """Create failure response with suggestions"""
        # Find fastest available combination (ignoring time limit)
        all_analyses = []
        for provider, model in model_combinations:
            if provider in self.config.providers:
                provider_config = self.config.providers[provider]
                if model in provider_config.models:
                    for api_key_id in provider_config.keys:
                        analysis = await self._analyze_combination(provider, model, api_key_id, 999999)
                        if analysis:
                            all_analyses.append(analysis)
        
        fastest = min(all_analyses, key=lambda x: x.total_seconds) if all_analyses else None
        
        retry_suggestion = None
        if fastest:
            available_at = datetime.now(timezone.utc) + timedelta(seconds=fastest.total_seconds)
            retry_suggestion = {
                "fastest_available_combination": {
                    "provider": fastest.provider,
                    "model": fastest.model,
                    "api_key": fastest.api_key_id,
                    "estimated_total_time_seconds": fastest.total_seconds,
                    "available_at": available_at.isoformat()
                }
            }
        
        evaluated_combinations = []
        for analysis in all_analyses:
            evaluated_combinations.append({
                "provider": analysis.provider,
                "model": analysis.model,
                "api_key": analysis.api_key_id,
                "total_estimated_time_seconds": analysis.total_seconds,
                "breakdown": {
                    "rate_limit_wait": analysis.rate_limit_wait_seconds,
                    "queue_wait": analysis.queue_wait_seconds,
                    "processing_time": analysis.processing_time_seconds
                }
            })
        
        return LLMResponse(
            success=False,
            request_id=request_id,
            error_code="no_combinations_available",
            message=f"No available combinations for {routing_method} routing",
            retry_suggestion=retry_suggestion,
            evaluated_combinations=evaluated_combinations
        )