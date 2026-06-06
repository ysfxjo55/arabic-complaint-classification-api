from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from configs.config import Settings
from configs.logging import get_logger
from interfaces.schemas.complaint import ComplaintResponse
from interfaces.schemas.explain import ClassificationExplainDetail
from services.llm_prompts import EXPLAIN_CLASSIFICATION_SYSTEM, EXPLAIN_CLASSIFICATION_USER_TEMPLATE

logger = get_logger("llm_service")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_json_fences(content: str) -> str:
    content = content.strip()
    m = _JSON_FENCE_RE.search(content)
    if m:
        return m.group(1).strip()
    return content


def _parse_explanation_json(raw: str) -> ClassificationExplainDetail:
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM JSON root must be an object")
    return ClassificationExplainDetail.model_validate(data)


async def explain_classification(
    *,
    text: str,
    classification: ComplaintResponse,
    settings: Settings,
) -> Tuple[Optional[ClassificationExplainDetail], Dict[str, Any], Optional[str]]:
    """
    Call OpenAI-compatible chat completions API. Returns (detail, meta, error_code).
    error_code is None on success.
    """
    if not settings.LLM_ENABLED:
        return None, {"explain_source": "disabled", "llm_latency_ms": None}, "LLM_DISABLED"

    if not settings.llm_configured():
        return None, {"explain_source": "disabled", "llm_latency_ms": None}, "MISSING_API_KEY"

    user = EXPLAIN_CLASSIFICATION_USER_TEMPLATE.format(
        original_text=text[:4000],
        sentiment_label=classification.sentiment.label.value,
        sentiment_confidence=classification.sentiment.confidence,
        topic_label=classification.topic.label.value,
        topic_confidence=classification.topic.confidence,
        intent_label=classification.intent.label.value,
        intent_confidence=classification.intent.confidence,
        action_label=classification.action.label,
        decision_source=classification.action.decision_source,
    )

    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": EXPLAIN_CLASSIFICATION_SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": settings.LLM_MAX_COMPLETION_TOKENS,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    url = f"{settings.LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    timeout = httpx.Timeout(settings.LLM_TIMEOUT_SECONDS, connect=10.0)
    started = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning("llm_timeout", latency_ms=elapsed_ms)
        return None, {"explain_source": "fallback", "llm_latency_ms": round(elapsed_ms, 2)}, "LLM_TIMEOUT"
    except httpx.RequestError as e:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning("llm_request_error", exc_info=True)
        return None, {
            "explain_source": "fallback",
            "llm_latency_ms": round(elapsed_ms, 2),
            "error_message": str(e),
        }, "LLM_HTTP_ERROR"

    elapsed_ms = (time.perf_counter() - started) * 1000
    meta: Dict[str, Any] = {
        "explain_source": "llm",
        "llm_latency_ms": round(elapsed_ms, 2),
        "llm_model": settings.LLM_MODEL,
    }

    if r.status_code >= 400:
        logger.warning(
            "llm_http_status",
            status_code=r.status_code,
            response_preview=(r.text or "")[:200],
        )
        meta["explain_source"] = "fallback"
        meta["http_status"] = r.status_code
        try:
            err_body = r.json()
            meta["error_body"] = err_body
        except Exception:
            meta["error_body"] = r.text[:500]
        return None, meta, "LLM_HTTP_ERROR"

    try:
        body = r.json()
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        logger.warning("llm_bad_envelope", exc_info=True)
        meta["explain_source"] = "fallback"
        return None, meta, "LLM_INVALID_RESPONSE"

    try:
        detail = _parse_explanation_json(content)
    except Exception:
        logger.warning("llm_invalid_json", exc_info=True)
        meta["explain_source"] = "fallback"
        meta["raw_content_preview"] = (content or "")[:400]
        return None, meta, "LLM_INVALID_RESPONSE"

    return detail, meta, None
