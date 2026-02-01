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
from ..utils.transaction_logger import get_transaction_logger, TransactionRecord
from ..utils.summary_stats import get_summary_stats_logger

logger = setup_logging()


class IntelligentRouter:
    """Core routing engine with intelligent combination selection - NOW SYNCHRONOUS"""

    def __init__(self, config_manager, rate_limit_manager, queue_manager, llm_client, circuit_breaker=None):
        self.config = config_manager
        self.rate_limits = rate_limit_manager
        self.queues = queue_manager
        self.llm_client = llm_client
        self.circuit_breaker = circuit_breaker

        # Import timeout manager
        from ..core.timeout_manager import GlobalTimeoutManager
        self.timeout_manager = GlobalTimeoutManager()

        logger.info("Router initialized with circuit breaker and timeout manager",
                   circuit_breaker_enabled=circuit_breaker.enabled if circuit_breaker else False)

    async def _log_transaction(self, record: TransactionRecord):
        """Log transaction to both detailed and summary loggers if available"""
        # Log to detailed transaction logger (CSV)
        transaction_logger = get_transaction_logger()
        if transaction_logger:
            await transaction_logger.log_transaction(record)

        # Log to summary stats logger (JSON daily aggregates)
        summary_logger = get_summary_stats_logger()
        if summary_logger:
            await summary_logger.log_request(
                success=record.success,
                routing_method=record.routing_method,
                intelligence_level=record.intelligence_level,
                provider_used=record.provider_used,
                error_type=record.error_type,
                cost_estimate=record.cost_estimate,
                total_time_ms=record.total_time_ms
            )

    async def _log_successful_transaction(self, transaction_record: TransactionRecord, full_response: LLMResponse,
                                        used_analysis, result: Dict[str, Any], attempt_info: Dict[str, Any], total_time: float):
        """Helper to log successful transaction"""
        if transaction_record:
            transaction_record.success = True
            transaction_record.provider_used = used_analysis.provider
            transaction_record.model_used = used_analysis.model
            transaction_record.api_key_used = used_analysis.api_key_id
            transaction_record.total_time_ms = int(total_time * 1000)
            transaction_record.decision_reason = full_response.decision_reason
            transaction_record.fallback_attempts = attempt_info.get('total_attempts', 1) - 1
            transaction_record.alternatives_tried = attempt_info.get('alternatives_tried', 1)

            # Extract usage and cost if available
            response_data = result.get("response", {})
            usage = response_data.get("usage")
            if usage:
                transaction_record.input_tokens = usage.get("input_tokens")
                transaction_record.output_tokens = usage.get("output_tokens")
                if transaction_record.input_tokens and transaction_record.output_tokens:
                    total_tokens = transaction_record.input_tokens + transaction_record.output_tokens
                    transaction_record.cost_estimate = (total_tokens / 1000) * used_analysis.cost_per_1k_tokens

            transaction_record.finish_reason = response_data.get("finish_reason")
            await self._log_transaction(transaction_record)

    async def _log_failed_transaction(self, transaction_record: TransactionRecord, error_type: str,
                                    error_message: str, total_time: float, attempt_info: Dict[str, Any] = None):
        """Helper to log failed transaction"""
        if transaction_record:
            transaction_record.success = False
            transaction_record.error_type = error_type
            transaction_record.error_message = error_message
            transaction_record.total_time_ms = int(total_time * 1000)
            if attempt_info:
                transaction_record.fallback_attempts = attempt_info.get('total_attempts', 1) - 1
                transaction_record.alternatives_tried = attempt_info.get('alternatives_tried', 1)
            await self._log_transaction(transaction_record)
    
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

        # Create transaction record
        transaction_logger = get_transaction_logger()
        transaction_record = None
        if transaction_logger:
            transaction_record = transaction_logger.create_record(
                request_id=request_id,
                client_id=request.client_id,
                routing_method="intelligence"
            )
            transaction_record.intelligence_level = request.intelligence_level.value
            transaction_record.max_wait_seconds = request.max_wait_seconds
        
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

            # Log transaction failure
            await self._log_failed_transaction(transaction_record, "no_models_available", full_response.message, time.time() - start_time)

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

        # Filter through circuit breaker
        if self.circuit_breaker and self.circuit_breaker.enabled:
            usable_analyses, blocked_analyses = await self.circuit_breaker.filter_alternatives(analyses)

            logger.info("🔌 CIRCUIT BREAKER: Filtered alternatives",
                       total=len(analyses),
                       usable=len(usable_analyses),
                       blocked=len(blocked_analyses))

            if not usable_analyses:
                full_response = LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="all_circuits_blocked",
                    message=f"All {len(analyses)} alternatives are circuit-blocked",
                    evaluated_combinations=blocked_analyses
                )
                await self._log_failed_transaction(transaction_record, "all_circuits_blocked", full_response.message, time.time() - start_time)

                if not request.debug:
                    return self._create_simplified_response(full_response)
                return full_response

            sorted_analyses = usable_analyses  # Already sorted by circuit breaker
        else:
            # Sort by best combination (cheapest, then fastest) if no circuit breaker
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
                sorted_analyses, request.messages, request.options, start_time
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

            # Log successful transaction
            await self._log_successful_transaction(transaction_record, full_response, used_analysis, result, attempt_info, total_time)

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

            # Log failed transaction
            await self._log_failed_transaction(transaction_record, "all_alternatives_failed", full_response.message, total_time, attempt_info)

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

        # Create transaction record
        transaction_logger = get_transaction_logger()
        transaction_record = None
        if transaction_logger:
            transaction_record = transaction_logger.create_record(
                request_id=request_id,
                client_id=request.client_id,
                routing_method="scenario"
            )
            transaction_record.scenario = request.scenario
            transaction_record.max_wait_seconds = request.max_wait_seconds
        
        # Get scenario combinations
        scenario_combinations = self.config.get_scenario_combinations(request.scenario)
        
        if not scenario_combinations:
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="scenario_not_found",
                message=f"Scenario not found: {request.scenario}"
            )

            # Log transaction failure
            await self._log_failed_transaction(transaction_record, "scenario_not_found", full_response.message, time.time() - start_time)

            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
        
        # Convert to (provider, model) tuples and analyze
        model_combinations = [(provider, model) for provider, model, _ in scenario_combinations]
        analyses = await self._analyze_all_combinations(model_combinations, request.max_wait_seconds)

        if not analyses:
            return await self._create_failure_response(model_combinations, request.max_wait_seconds, "scenario", request_id)

        # Filter through circuit breaker
        if self.circuit_breaker and self.circuit_breaker.enabled:
            usable_analyses, blocked_analyses = await self.circuit_breaker.filter_alternatives(analyses)

            logger.info("🔌 CIRCUIT BREAKER: Filtered scenario alternatives",
                       total=len(analyses),
                       usable=len(usable_analyses),
                       blocked=len(blocked_analyses))

            if not usable_analyses:
                full_response = LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="all_circuits_blocked",
                    message=f"All {len(analyses)} scenario alternatives are circuit-blocked",
                    evaluated_combinations=blocked_analyses
                )
                await self._log_failed_transaction(transaction_record, "all_circuits_blocked", full_response.message, time.time() - start_time)

                if not request.debug:
                    return self._create_simplified_response(full_response)
                return full_response

            analyses = usable_analyses

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
            sorted_analyses, request.messages, request.options, start_time
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

            # Log successful transaction
            await self._log_successful_transaction(transaction_record, full_response, used_analysis, result, attempt_info, total_time)

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

            # Log failed transaction
            await self._log_failed_transaction(transaction_record, "all_alternatives_failed", full_response.message, total_time, attempt_info)

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

        # Create transaction record
        transaction_logger = get_transaction_logger()
        transaction_record = None
        if transaction_logger:
            transaction_record = transaction_logger.create_record(
                request_id=request_id,
                client_id=request.client_id,
                routing_method="direct"
            )
            transaction_record.requested_provider = request.provider
            transaction_record.requested_model = request.model_name
            transaction_record.max_wait_seconds = request.max_wait_seconds
        
        # Validate provider and model
        if request.provider not in self.config.providers:
            full_response = LLMResponse(
                success=False,
                request_id=request_id,
                error_code="provider_not_found",
                message=f"Provider not found: {request.provider}"
            )

            # Log transaction failure
            await self._log_failed_transaction(transaction_record, "provider_not_found", full_response.message, time.time() - start_time)

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

            # Log transaction failure
            await self._log_failed_transaction(transaction_record, "model_not_found", full_response.message, time.time() - start_time)

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

                # Log transaction failure
                await self._log_failed_transaction(transaction_record, "api_key_not_found", full_response.message, time.time() - start_time)

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

        # Filter through circuit breaker
        if self.circuit_breaker and self.circuit_breaker.enabled:
            usable_analyses, blocked_analyses = await self.circuit_breaker.filter_alternatives(analyses)

            logger.info("🔌 CIRCUIT BREAKER: Filtered direct alternatives",
                       total=len(analyses),
                       usable=len(usable_analyses),
                       blocked=len(blocked_analyses))

            if not usable_analyses:
                full_response = LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="all_circuits_blocked",
                    message=f"All {len(analyses)} direct alternatives are circuit-blocked",
                    evaluated_combinations=blocked_analyses
                )
                await self._log_failed_transaction(transaction_record, "all_circuits_blocked", full_response.message, time.time() - start_time)

                if not request.debug:
                    return self._create_simplified_response(full_response)
                return full_response

            sorted_analyses = usable_analyses  # Already prioritized by circuit breaker
        else:
            # Sort by fastest completion time for direct requests if no circuit breaker
            sorted_analyses = sorted(analyses, key=lambda x: x.total_seconds)
        
        logger.info("Attempting direct request with retry logic",
                   provider=request.provider,
                   model=request.model_name,
                   alternatives_count=len(sorted_analyses))

        # Try each alternative with retry logic
        result, used_analysis, attempt_info = await self._execute_with_retry_and_fallback(
            sorted_analyses, request.messages, request.options, start_time
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

            # Log successful transaction
            await self._log_successful_transaction(transaction_record, full_response, used_analysis, result, attempt_info, total_time)

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

            # Log failed transaction
            await self._log_failed_transaction(transaction_record, "all_alternatives_failed", full_response.message, total_time, attempt_info)

            if not request.debug:
                return self._create_simplified_response(full_response)
            return full_response
    
    async def _execute_with_retry_and_fallback(self, sorted_analyses: List[CompletionTimeAnalysis],
                                             messages: List[Message], options: Optional[LLMOptions],
                                             start_time: float = None) -> Tuple[Dict[str, Any], CompletionTimeAnalysis, Dict[str, Any]]:
        """Execute request with retry logic and automatic failover - enhanced error handling with 429 provider awareness"""
        MAX_RETRIES = 3
        attempt_info = {
            "alternatives_tried": 0,
            "total_attempts": 0,
            "failures": []
        }

        # Track start time for global timeout
        if start_time is None:
            start_time = time.time()

        escalated = False  # Track if we've escalated to fast mode
        current_alternatives = sorted_analyses[:]
        rate_limited_providers = set()  # Track providers that hit 429
        last_tried_provider = None

        while current_alternatives:
            # Check global timeout
            if self.timeout_manager.should_abort(start_time):
                elapsed = time.time() - start_time
                self.timeout_manager.log_timeout_abort(elapsed, attempt_info["alternatives_tried"])

                return {
                    "success": False,
                    "error": f"Hard timeout exceeded ({elapsed:.1f}s) after {attempt_info['alternatives_tried']} alternatives"
                }, sorted_analyses[0] if sorted_analyses else None, attempt_info

            # Check if we should escalate to fast mode
            if not escalated and self.timeout_manager.should_escalate(start_time):
                escalated = True
                elapsed = time.time() - start_time
                self.timeout_manager.log_escalation(elapsed, attempt_info["alternatives_tried"])

                # Reorder alternatives for speed (different provider, CLOSED circuits only)
                if last_tried_provider:
                    current_alternatives = self.timeout_manager.reorder_for_escalation(
                        current_alternatives,
                        last_tried_provider
                    )

                # Reduce MAX_RETRIES in escalated mode
                MAX_RETRIES = 1

                logger.warning("⏱️ ESCALATED TO FAST MODE",
                             elapsed_seconds=elapsed,
                             remaining_alternatives=len(current_alternatives),
                             max_retries_reduced=MAX_RETRIES)
            # Get next alternative
            analysis = current_alternatives.pop(0)
            attempt_info["alternatives_tried"] += 1
            last_tried_provider = analysis.provider

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

                        # Record success with circuit breaker
                        if self.circuit_breaker and self.circuit_breaker.enabled:
                            response_time_ms = result.get("response_time_ms")
                            await self.circuit_breaker.record_success(
                                analysis.provider,
                                analysis.model,
                                analysis.api_key_id,
                                response_time_ms
                            )

                        return result, analysis, attempt_info
                    else:
                        # Check error handling strategy
                        error_msg = result.get("error", "Unknown error")
                        status_code = result.get("status_code")
                        response_time_ms = result.get("response_time_ms")

                        # Record failure with circuit breaker
                        if self.circuit_breaker and self.circuit_breaker.enabled:
                            should_skip_alt, should_skip_prov = await self.circuit_breaker.record_failure(
                                analysis.provider,
                                analysis.model,
                                analysis.api_key_id,
                                error_msg,
                                status_code,
                                response_time_ms,
                                result.get("headers")
                            )

                            # Use circuit breaker's strategy
                            if should_skip_prov:
                                error_strategy = "skip_provider"
                            elif should_skip_alt:
                                error_strategy = "skip_alternative"
                            else:
                                error_strategy = "retry"
                        else:
                            # Fallback to old strategy if circuit breaker disabled
                            error_strategy = self._should_skip_retries(error_msg)

                        logger.warning("Attempt failed",
                                     alternative=attempt_info["alternatives_tried"],
                                     retry=retry_attempt,
                                     provider=analysis.provider,
                                     model=analysis.model,
                                     error=error_msg,
                                     strategy=error_strategy)

                        attempt_info["failures"].append({
                            "alternative": attempt_info["alternatives_tried"],
                            "retry": retry_attempt,
                            "provider": analysis.provider,
                            "model": analysis.model,
                            "api_key_id": analysis.api_key_id,
                            "error": error_msg,
                            "strategy": error_strategy
                        })
                        
                        # Handle different error strategies
                        if error_strategy == "skip_alternative":
                            # For rate limiting (429), reorder remaining alternatives to prioritize same provider
                            if "429" in error_msg or "rate limited" in error_msg.lower():
                                rate_limited_providers.add(analysis.provider)
                                current_alternatives = self._reorder_alternatives_for_rate_limit(
                                    current_alternatives, analysis.provider
                                )
                                logger.info("Rate limit detected, reordered alternatives for provider recovery", 
                                           provider=analysis.provider,
                                           remaining_alternatives=len(current_alternatives))
                            break  # Skip to next alternative
                        elif error_strategy == "skip_provider":
                            # Remove all remaining alternatives from this provider
                            provider_to_skip = analysis.provider
                            original_count = len(current_alternatives)
                            current_alternatives = [alt for alt in current_alternatives if alt.provider != provider_to_skip]
                            removed_count = original_count - len(current_alternatives)
                            logger.info("Provider authentication failed, removing all alternatives", 
                                       provider=provider_to_skip,
                                       removed_alternatives=removed_count,
                                       remaining_alternatives=len(current_alternatives))
                            break  # Skip to next alternative
                        elif error_strategy == "retry":
                            # Continue with retry logic
                            if retry_attempt == MAX_RETRIES:
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

                    # Record failure with circuit breaker
                    if self.circuit_breaker and self.circuit_breaker.enabled:
                        should_skip_alt, should_skip_prov = await self.circuit_breaker.record_failure(
                            analysis.provider,
                            analysis.model,
                            analysis.api_key_id,
                            error_msg,
                            status_code=None,
                            response_time_ms=None,
                            response_headers=None
                        )

                        # Use circuit breaker's strategy
                        if should_skip_prov:
                            error_strategy = "skip_provider"
                        elif should_skip_alt:
                            error_strategy = "skip_alternative"
                        else:
                            error_strategy = "retry"
                    else:
                        # Fallback to old strategy if circuit breaker disabled
                        error_strategy = self._should_skip_retries(str(e))

                    logger.error("Unexpected error during attempt",
                               alternative=attempt_info["alternatives_tried"],
                               retry=retry_attempt,
                               provider=analysis.provider,
                               model=analysis.model,
                               error=error_msg,
                               strategy=error_strategy,
                               traceback=traceback.format_exc())

                    attempt_info["failures"].append({
                        "alternative": attempt_info["alternatives_tried"],
                        "retry": retry_attempt,
                        "provider": analysis.provider,
                        "model": analysis.model,
                        "api_key_id": analysis.api_key_id,
                        "error": error_msg,
                        "strategy": error_strategy
                    })
                    
                    # Apply same strategy logic as above
                    if error_strategy != "retry" or retry_attempt == MAX_RETRIES:
                        break
                    else:
                        wait_time = min(2 ** (retry_attempt - 1), 5)
                        await asyncio.sleep(wait_time)
        
        # All alternatives exhausted
        last_error = attempt_info["failures"][-1]["error"] if attempt_info["failures"] else "No alternatives available"
        logger.error("All alternatives exhausted", 
                   alternatives_tried=attempt_info["alternatives_tried"],
                   total_attempts=attempt_info["total_attempts"],
                   rate_limited_providers=list(rate_limited_providers),
                   last_error=last_error)
        
        return {
            "success": False,
            "error": f"All {attempt_info['alternatives_tried']} alternatives failed after {attempt_info['total_attempts']} total attempts"
        }, sorted_analyses[0] if sorted_analyses else None, attempt_info
    
    def _should_skip_retries(self, error_msg: str) -> str:
        """Determine error handling strategy: 'retry', 'skip_alternative', or 'skip_provider'"""
        error_lower = error_msg.lower()
        
        # Rate limiting errors - skip to next alternative within same provider first
        if any(phrase in error_lower for phrase in [
            "rate limited", "429", "quota", "exceeded", "resource_exhausted"
        ]):
            return "skip_alternative"
            
        # Authentication errors - skip entire provider (all keys likely invalid)
        if any(phrase in error_lower for phrase in [
            "401", "403", "unauthorized", "forbidden", "invalid api key"
        ]):
            return "skip_provider"
            
        # Content/response errors that won't be fixed by retrying - skip this alternative
        if any(phrase in error_lower for phrase in [
            "empty content", "could not extract content", "max_tokens"
        ]):
            return "skip_alternative"
            
        # Network errors that might be temporary - allow retries
        if any(phrase in error_lower for phrase in [
            "timeout", "connection", "network", "500", "502", "503", "504"
        ]):
            return "retry"
            
        # Default: allow retries for unknown errors
        return "retry"
    
    def _reorder_alternatives_for_rate_limit(self, sorted_analyses: List[CompletionTimeAnalysis], 
                                           failed_provider: str) -> List[CompletionTimeAnalysis]:
        """Reorder alternatives to prioritize same provider alternatives after 429 error"""
        # Separate alternatives by provider
        same_provider = []
        other_providers = []
        
        for analysis in sorted_analyses:
            if analysis.provider == failed_provider:
                same_provider.append(analysis)
            else:
                other_providers.append(analysis)
        
        # For same provider alternatives, sort by: different models first, then different API keys
        # This handles the case where rate limits might be per-model or per-key
        same_provider_reordered = []
        processed_models = set()
        
        # First pass: different models with same provider (rate limits often per-model)
        for analysis in same_provider:
            if analysis.model not in processed_models:
                same_provider_reordered.append(analysis)
                processed_models.add(analysis.model)
        
        # Second pass: same models but different API keys (rate limits often per-key)
        for analysis in same_provider:
            if analysis not in same_provider_reordered:
                same_provider_reordered.append(analysis)
        
        # Return: same provider alternatives first, then other providers
        logger.info("🔄 Reordered alternatives for rate limit recovery", 
                   failed_provider=failed_provider,
                   same_provider_count=len(same_provider_reordered),
                   other_provider_count=len(other_providers),
                   reordered_sequence=[f"{a.provider}/{a.model}/{a.api_key_id}" for a in same_provider_reordered + other_providers][:5])
        
        return same_provider_reordered + other_providers
    
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
            
            # Special handling for Ollama: check health before proceeding
            if provider == "ollama":
                is_healthy = await self.llm_client.check_ollama_health(model_config.endpoint)
                if not is_healthy:
                    logger.warning("Ollama health check failed, skipping", 
                                 provider=provider, model=model, endpoint=model_config.endpoint)
                    return None
            
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

    # Vision/Multimodal Routing Methods

    async def _preprocess_vision_messages(self, messages: List) -> List[Dict]:
        """
        Preprocess vision messages: fetch URLs, validate images

        Converts all image URLs to base64 for consistent provider handling.
        Validates all base64 images for format, size, and dimensions.

        Returns:
            List of preprocessed messages in dict format
        """
        from ..utils.image_processor import ImageFetcher, ImageValidator

        processed = []
        # Reuse the existing session from llm_client instead of creating a new one
        if not self.llm_client.session:
            raise RuntimeError("LLM client session not initialized")

        session = self.llm_client.session
        for msg in messages:
            content_blocks = []
            for block in msg.content:
                if block.type == "text":
                    content_blocks.append({
                        "type": "text",
                        "text": block.text
                    })
                elif block.type == "image":
                    if block.source.type == "url":
                        # Fetch URL and convert to base64
                        base64_data, media_type, error = await ImageFetcher.fetch_url_to_base64(
                            block.source.url, session
                        )
                        if error:
                            raise ValueError(f"Image fetch failed for {block.source.url}: {error}")

                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data
                            }
                        })
                    else:  # base64
                        # Validate base64 image
                        valid, error = await ImageValidator.validate_base64_image(
                            block.source.data, block.source.media_type
                        )
                        if not valid:
                            raise ValueError(f"Image validation failed: {error}")

                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.source.media_type,
                                "data": block.source.data
                            }
                        })

            processed.append({
                "role": msg.role,
                "content": content_blocks
            })

        return processed

    def _get_vision_models_by_intelligence(self, intelligence_level: str) -> List[tuple]:
        """
        Get vision-capable models for intelligence level

        Returns:
            List of (provider, model) tuples that support vision
        """
        combinations = []
        for provider, model in self.config.get_models_by_intelligence(intelligence_level):
            if provider in self.config.providers:
                provider_config = self.config.providers[provider]
                if model in provider_config.models:
                    model_config = provider_config.models[model]

                    # Check if model supports vision
                    if getattr(model_config, 'supports_vision', False):
                        combinations.append((provider, model))

        return combinations

    def _validate_vision_request(self, messages: List[Dict], model_config) -> tuple:
        """
        Validate vision request against model capabilities

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not getattr(model_config, 'supports_vision', False):
            return False, "Model does not support vision capabilities"

        # Count images in messages
        image_count = 0
        for msg in messages:
            for block in msg.get("content", []):
                if block.get("type") == "image":
                    image_count += 1

        # Check for empty image list (vision request should have at least one image)
        if image_count == 0:
            logger.warning("Vision request contains no images - this may not be intentional")
            # Don't fail validation, just warn - some models can handle text-only in vision mode

        # Check vision limits
        vision_limits = getattr(model_config, 'vision_limits', None)
        if vision_limits:
            max_images = getattr(vision_limits, 'max_images_per_request', 1)
            if image_count > max_images:
                return False, f"Image count {image_count} exceeds model limit {max_images}"

        return True, None

    async def route_vision_intelligence_request(self, request) -> LLMResponse:
        """Route vision request based on intelligence level"""
        request_id = str(uuid.uuid4())
        logger.info("Vision intelligence routing started",
                   request_id=request_id,
                   intelligence_level=request.intelligence_level.value)

        try:
            # Preprocess images (fetch URLs, validate)
            logger.info("Preprocessing vision messages", request_id=request_id)
            processed_messages = await self._preprocess_vision_messages(request.messages)

            # Get vision-capable models for intelligence level
            vision_models = self._get_vision_models_by_intelligence(request.intelligence_level.value)

            if not vision_models:
                logger.warning("No vision models available",
                             intelligence_level=request.intelligence_level.value)
                return LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="no_vision_models_available",
                    message=f"No vision-capable models for intelligence level: {request.intelligence_level.value}"
                )

            logger.info("Found vision models", count=len(vision_models), models=vision_models)

            # Analyze all vision-capable combinations
            analyses = []
            for provider, model in vision_models:
                if provider in self.config.providers:
                    provider_config = self.config.providers[provider]
                    model_config = provider_config.models.get(model)

                    # Validate vision request
                    valid, error = self._validate_vision_request(processed_messages, model_config)
                    if not valid:
                        logger.debug("Vision validation failed",
                                   provider=provider, model=model, error=error)
                        continue

                    for api_key_id in provider_config.keys:
                        analysis = await self._analyze_combination(
                            provider, model, api_key_id, request.max_wait_seconds
                        )
                        if analysis:
                            analyses.append(analysis)

            if not analyses:
                return await self._create_failure_response(
                    vision_models, request.max_wait_seconds, "intelligence", request_id
                )

            # Sort by total completion time
            analyses.sort(key=lambda x: (x.priority_penalty, x.total_seconds))
            selected = analyses[0]

            logger.info("Selected vision combination",
                       provider=selected.provider,
                       model=selected.model,
                       total_time=selected.total_seconds)

            # Execute vision request
            result = await self._execute_llm_request_vision(
                selected, request_id, processed_messages,
                request.options.model_dump() if request.options else {},
                request.debug
            )

            return result

        except Exception as e:
            logger.error("Vision intelligence routing failed",
                        request_id=request_id, error=str(e))
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="vision_routing_error",
                message=f"Vision routing failed: {str(e)}"
            )

    async def route_vision_scenario_request(self, request) -> LLMResponse:
        """Route vision request based on scenario"""
        request_id = str(uuid.uuid4())
        logger.info("Vision scenario routing started",
                   request_id=request_id,
                   scenario=request.scenario)

        try:
            # Preprocess images
            processed_messages = await self._preprocess_vision_messages(request.messages)

            # Get scenario model priorities
            if request.scenario not in self.config.routing_scenarios:
                return LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="scenario_not_found",
                    message=f"Scenario '{request.scenario}' not found in routing.json"
                )

            scenario_priorities = self.config.routing_scenarios[request.scenario]

            # Filter for vision-capable models only
            vision_combinations = []
            for provider, model in scenario_priorities:
                if provider in self.config.providers:
                    provider_config = self.config.providers[provider]
                    if model in provider_config.models:
                        model_config = provider_config.models[model]
                        if getattr(model_config, 'supports_vision', False):
                            vision_combinations.append((provider, model))

            if not vision_combinations:
                return LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="no_vision_models_in_scenario",
                    message=f"No vision-capable models in scenario '{request.scenario}'"
                )

            # Analyze vision-capable combinations
            analyses = []
            for provider, model in vision_combinations:
                provider_config = self.config.providers[provider]
                model_config = provider_config.models[model]

                # Validate vision request
                valid, error = self._validate_vision_request(processed_messages, model_config)
                if not valid:
                    continue

                for api_key_id in provider_config.keys:
                    analysis = await self._analyze_combination(
                        provider, model, api_key_id, request.max_wait_seconds
                    )
                    if analysis:
                        analyses.append(analysis)

            if not analyses:
                return await self._create_failure_response(
                    vision_combinations, request.max_wait_seconds, "scenario", request_id
                )

            # Sort by total completion time
            analyses.sort(key=lambda x: (x.priority_penalty, x.total_seconds))
            selected = analyses[0]

            # Execute vision request
            result = await self._execute_llm_request_vision(
                selected, request_id, processed_messages,
                request.options.model_dump() if request.options else {},
                request.debug
            )

            return result

        except Exception as e:
            logger.error("Vision scenario routing failed",
                        request_id=request_id, error=str(e))
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="vision_scenario_error",
                message=f"Vision scenario routing failed: {str(e)}"
            )

    async def route_vision_direct_request(self, request) -> LLMResponse:
        """Route vision request to specific provider/model"""
        request_id = str(uuid.uuid4())
        logger.info("Vision direct routing started",
                   request_id=request_id,
                   provider=request.provider,
                   model=request.model_name)

        try:
            # Preprocess images
            processed_messages = await self._preprocess_vision_messages(request.messages)

            # Validate provider and model
            if request.provider not in self.config.providers:
                return LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="provider_not_found",
                    message=f"Provider '{request.provider}' not configured"
                )

            provider_config = self.config.providers[request.provider]
            if request.model_name not in provider_config.models:
                return LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="model_not_found",
                    message=f"Model '{request.model_name}' not found for provider '{request.provider}'"
                )

            model_config = provider_config.models[request.model_name]

            # Validate vision support
            valid, error = self._validate_vision_request(processed_messages, model_config)
            if not valid:
                return LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="vision_not_supported",
                    message=error
                )

            # Select API key
            if request.api_key_id:
                if request.api_key_id not in provider_config.keys:
                    return LLMResponse(
                        success=False,
                        request_id=request_id,
                        error_code="api_key_not_found",
                        message=f"API key '{request.api_key_id}' not found"
                    )
                api_key_id = request.api_key_id
            else:
                # Auto-select first available key
                api_key_id = list(provider_config.keys.keys())[0]

            # Analyze combination
            analysis = await self._analyze_combination(
                request.provider, request.model_name, api_key_id, request.max_wait_seconds
            )

            if not analysis:
                return LLMResponse(
                    success=False,
                    request_id=request_id,
                    error_code="combination_unavailable",
                    message="Requested combination is not available"
                )

            # Execute vision request
            result = await self._execute_llm_request_vision(
                analysis, request_id, processed_messages,
                request.options.model_dump() if request.options else {},
                request.debug
            )

            return result

        except Exception as e:
            logger.error("Vision direct routing failed",
                        request_id=request_id, error=str(e))
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="vision_direct_error",
                message=f"Vision direct routing failed: {str(e)}"
            )

    async def _execute_llm_request_vision(self, analysis: CompletionTimeAnalysis,
                                        request_id: str, messages: List[Dict],
                                        options: Dict, debug: bool) -> LLMResponse:
        """Execute vision LLM request using selected combination"""
        start_time = datetime.now(timezone.utc)

        try:
            # Get configuration
            provider_config = self.config.providers[analysis.provider]
            model_config = provider_config.models[analysis.model]
            key_config = provider_config.keys[analysis.api_key_id]

            # Wait for rate limits if needed
            if analysis.rate_limit_wait_seconds > 0:
                await asyncio.sleep(analysis.rate_limit_wait_seconds)

            # Wait for queue if needed
            if analysis.queue_wait_seconds > 0:
                await asyncio.sleep(analysis.queue_wait_seconds)

            # Make vision API call
            api_start = datetime.now(timezone.utc)
            result = await self.llm_client._make_vision_api_call(
                provider=analysis.provider,
                model=analysis.model,
                api_key=key_config.api_key,
                messages=messages,
                options=options,
                endpoint=model_config.endpoint,
                version=model_config.version if hasattr(model_config, 'version') else None
            )
            api_end = datetime.now(timezone.utc)

            # Calculate metrics
            api_response_time_ms = int((api_end - api_start).total_seconds() * 1000)
            total_time_ms = int((api_end - start_time).total_seconds() * 1000)

            # Calculate cost
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cost_usd = ((input_tokens + output_tokens) / 1000) * model_config.cost_per_1k_tokens

            # Build response
            metadata = ResponseMetadata(
                rate_limit_wait_ms=int(analysis.rate_limit_wait_seconds * 1000),
                queue_wait_ms=int(analysis.queue_wait_seconds * 1000),
                api_response_time_ms=api_response_time_ms,
                total_completion_time_ms=total_time_ms,
                cost_usd=cost_usd,
                alternatives_considered=[]
            )

            return LLMResponse(
                success=True,
                request_id=request_id,
                provider_used=analysis.provider,
                model_used=analysis.model,
                api_key_used=analysis.api_key_id,
                routing_method=RoutingMethod.INTELLIGENCE,
                decision_reason=f"Vision-capable model selected",
                response=result,
                metadata=metadata
            )

        except Exception as e:
            logger.error("Vision LLM request execution failed",
                        request_id=request_id,
                        provider=analysis.provider,
                        model=analysis.model,
                        error=str(e))
            return LLMResponse(
                success=False,
                request_id=request_id,
                error_code="vision_execution_error",
                message=f"Vision request execution failed: {str(e)}"
            )