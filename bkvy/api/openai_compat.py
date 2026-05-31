"""
OpenAI-compatible API surface for bkvy.

A thin plumbing/translation layer on top of the existing intelligence router.
It does NOT change any core routing logic — it only:
  1. accepts the OpenAI Chat Completions request shape,
  2. maps it onto an internal IntelligenceRequest,
  3. calls the existing router.route_intelligence_request(),
  4. reshapes the result into an OpenAI ChatCompletion response.

The intelligence tier (low | medium | high) is selected two ways:
  * POST /v1/chat/completions       with  "model": "low" | "medium" | "high"
                                          (also accepts "bkvy-low", etc.)
  * POST /v1/{tier}/chat/completions  e.g. /v1/low/chat/completions
    (here the JSON "model" field is ignored)

Non-streaming only — stream=true returns a 400.
"""

import os
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..models.enums import IntelligenceLevel
from ..models.schemas import IntelligenceRequest, Message, LLMOptions
from .lifespan import get_config_manager, get_router

router = APIRouter(tags=["openai-compatible"])

# How long the router may spend (estimate budget) per request. OpenAI clients
# have no equivalent field, so it comes from config with a sane default.
DEFAULT_MAX_WAIT_SECONDS = int(os.getenv("OPENAI_COMPAT_MAX_WAIT", "300"))

VALID_TIERS = {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Request models (lenient — we accept and ignore extra OpenAI fields)
# ---------------------------------------------------------------------------
class OpenAIMessage(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")
    role: str
    # OpenAI allows content to be a string OR a list of content parts; allow both.
    content: Union[str, List[Dict[str, Any]], None] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    user: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_tier(raw: Optional[str]) -> Optional[str]:
    """Map a model field to a tier string, or None if unrecognized."""
    if not raw:
        return None
    name = raw.strip().lower()
    # strip common prefixes: "bkvy-low", "bkvy/low", "intelligence:high"
    for sep in ("-", "/", ":", "_"):
        if name.startswith(f"bkvy{sep}"):
            name = name[len(f"bkvy{sep}"):]
    if name in VALID_TIERS:
        return name
    return None


def _flatten_content(content: Union[str, List[Dict[str, Any]], None]) -> str:
    """Coerce OpenAI message content (string or content-part list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # list of parts: keep text parts, join. Non-text parts are dropped (this
    # layer is text-only; the underlying router has its own multimodal path).
    parts = []
    for p in content:
        if isinstance(p, dict):
            if p.get("type") == "text" and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p.get("text"), str):
                parts.append(p["text"])
    return "\n".join(parts)


def _map_finish_reason(simplified_finish: Optional[str], truncated: Optional[bool]) -> str:
    """Map provider-specific finish reasons to OpenAI's vocabulary."""
    if truncated:
        return "length"
    if not simplified_finish:
        return "stop"
    f = simplified_finish.lower()
    if f in ("max_tokens", "length"):
        return "length"
    if f in ("stop", "end_turn", "stop_sequence", "complete"):
        return "stop"
    if f in ("content_filter", "safety"):
        return "content_filter"
    return "stop"


def _build_options(req: ChatCompletionRequest) -> Optional[LLMOptions]:
    stop = req.stop
    if isinstance(stop, str):
        stop = [stop]
    opts = LLMOptions(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=stop,
    )
    # Only attach options if at least one field is set.
    if any(v is not None for v in (opts.max_tokens, opts.temperature, opts.top_p, opts.stop)):
        return opts
    return None


def _error_response(message: str, status_code: int, err_type: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "code": code}},
    )


async def _handle_chat_completion(tier: str, req: ChatCompletionRequest) -> JSONResponse:
    """Shared handler: translate -> route -> reshape into OpenAI format."""
    if req.stream:
        return _error_response(
            "Streaming is not supported by this endpoint. Set stream=false.",
            400, "invalid_request_error", "streaming_not_supported",
        )

    if not req.messages:
        return _error_response(
            "'messages' must contain at least one message.",
            400, "invalid_request_error", "missing_messages",
        )

    internal_messages = [
        Message(role=m.role, content=_flatten_content(m.content)) for m in req.messages
    ]

    intelligence_request = IntelligenceRequest(
        client_id=req.user or "openai-compat",
        intelligence_level=IntelligenceLevel(tier),
        max_wait_seconds=DEFAULT_MAX_WAIT_SECONDS,
        messages=internal_messages,
        options=_build_options(req),
        debug=False,
    )

    config_manager = get_config_manager()
    bkvy_router = get_router()
    if bkvy_router is None:
        return _error_response(
            "Router is not initialized.", 503, "server_error", "router_unavailable",
        )

    # Keep config hot-reload behavior consistent with the native endpoint.
    await config_manager.refresh_if_changed()

    result = await bkvy_router.route_intelligence_request(intelligence_request)

    # route_intelligence_request returns a SimplifiedResponse (debug=False).
    success = getattr(result, "success", False)
    request_id = getattr(result, "request_id", "unknown")

    if not success:
        err_msg = getattr(result, "error", None) or getattr(result, "message", None) \
            or "All upstream model alternatives failed."
        return _error_response(
            err_msg, 502, "upstream_error", "all_alternatives_failed",
        )

    usage = getattr(result, "usage", None) or {}
    content = getattr(result, "content", None) or ""
    model_used = getattr(result, "model_used", None) or f"bkvy-{tier}"
    finish_reason = _map_finish_reason(
        getattr(result, "finish_reason", None), getattr(result, "truncated", None)
    )

    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

    return JSONResponse(
        status_code=200,
        content={
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_used,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """OpenAI-compatible endpoint. Tier is taken from the 'model' field."""
    tier = _normalize_tier(req.model)
    if tier is None:
        return _error_response(
            f"Unknown model '{req.model}'. Use one of: low, medium, high "
            f"(or bkvy-low / bkvy-medium / bkvy-high), or POST to /v1/{{tier}}/chat/completions.",
            400, "invalid_request_error", "model_not_found",
        )
    return await _handle_chat_completion(tier, req)


@router.post("/v1/{tier}/chat/completions")
async def chat_completions_per_tier(
    req: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    tier: str = Path(..., description="Intelligence tier: low, medium, or high"),
):
    """Per-tier OpenAI-compatible endpoint. The JSON 'model' field is ignored."""
    norm = _normalize_tier(tier)
    if norm is None:
        return _error_response(
            f"Unknown tier '{tier}'. Use low, medium, or high.",
            404, "invalid_request_error", "tier_not_found",
        )
    return await _handle_chat_completion(norm, req)


@router.get("/v1/models")
async def list_models():
    """List the intelligence tiers as OpenAI-style model objects."""
    created = int(time.time())
    data = [
        {"id": tier, "object": "model", "created": created, "owned_by": "bkvy"}
        for tier in ("low", "medium", "high")
    ]
    return {"object": "list", "data": data}
