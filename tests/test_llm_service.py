import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from configs.config import Settings
from interfaces.schemas.complaint import ActionDetail, ComplaintResponse, PredictionDetail
from interfaces.schemas.enums import ActionLabel, SentimentLabel, TopicLabel
from services import llm_service


def _classification() -> ComplaintResponse:
    return ComplaintResponse(
        sentiment=PredictionDetail(
            label=SentimentLabel.NEG, confidence=0.9, explanation="x"
        ),
        topic=PredictionDetail(label=TopicLabel.TECH, confidence=0.8, explanation="x"),
        intent=PredictionDetail(
            label=ActionLabel.REPORT_BUG, confidence=0.7, explanation="x"
        ),
        action=ActionDetail(label="A", decision_source="RULE_ENGINE"),
        meta={},
    )


def _settings_with_key() -> Settings:
    s = Settings()
    s.LLM_ENABLED = True
    s.OPENAI_API_KEY = "sk-test"
    s.LLM_BASE_URL = "https://api.openai.com/v1"
    s.LLM_MODEL = "gpt-4o-mini"
    s.LLM_TIMEOUT_SECONDS = 5.0
    s.LLM_MAX_COMPLETION_TOKENS = 256
    return s


async def _run(coro):
    return await coro


def test_llm_disabled_returns_code():
    s = Settings()
    s.LLM_ENABLED = False

    async def go():
        d, m, err = await llm_service.explain_classification(
            text="t", classification=_classification(), settings=s
        )
        assert d is None
        assert err == "LLM_DISABLED"
        assert m["explain_source"] == "disabled"

    asyncio.run(go())


def test_missing_key_returns_code():
    s = Settings()
    s.LLM_ENABLED = True
    s.OPENAI_API_KEY = None

    async def go():
        d, m, err = await llm_service.explain_classification(
            text="t", classification=_classification(), settings=s
        )
        assert err == "MISSING_API_KEY"

    asyncio.run(go())


def test_llm_success_parses_json():
    s = _settings_with_key()
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "summary": "S",
                            "rationale": "R",
                            "limitations": "L",
                        }
                    )
                }
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = payload

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def go():
        with patch("services.llm_service.httpx.AsyncClient", return_value=mock_client):
            d, m, err = await llm_service.explain_classification(
                text="complaint", classification=_classification(), settings=s
            )
        assert err is None
        assert d is not None
        assert d.summary == "S"
        assert m["explain_source"] == "llm"
        assert "llm_latency_ms" in m

    asyncio.run(go())


def test_llm_timeout():
    s = _settings_with_key()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def go():
        with patch("services.llm_service.httpx.AsyncClient", return_value=mock_client):
            d, m, err = await llm_service.explain_classification(
                text="t", classification=_classification(), settings=s
            )
        assert d is None
        assert err == "LLM_TIMEOUT"
        assert m["explain_source"] == "fallback"

    asyncio.run(go())


def test_llm_http_error():
    s = _settings_with_key()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "err"
    mock_resp.json.side_effect = ValueError()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def go():
        with patch("services.llm_service.httpx.AsyncClient", return_value=mock_client):
            d, m, err = await llm_service.explain_classification(
                text="t", classification=_classification(), settings=s
            )
        assert d is None
        assert err == "LLM_HTTP_ERROR"

    asyncio.run(go())


def test_llm_invalid_json_content():
    s = _settings_with_key()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "not json"}}]
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def go():
        with patch("services.llm_service.httpx.AsyncClient", return_value=mock_client):
            d, m, err = await llm_service.explain_classification(
                text="t", classification=_classification(), settings=s
            )
        assert d is None
        assert err == "LLM_INVALID_RESPONSE"
        assert m["explain_source"] == "fallback"

    asyncio.run(go())


def test_strip_json_fences():
    raw = '```json\n{"summary":"a","rationale":"b","limitations":""}\n```'
    d = llm_service._parse_explanation_json(raw)
    assert d.summary == "a"
