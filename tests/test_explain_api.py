from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from interfaces.api.explain_route import router as explain_router
from interfaces.schemas.complaint import (
    ActionDetail,
    ComplaintResponse,
    PredictionDetail,
)
from interfaces.schemas.enums import ActionLabel, SentimentLabel, TopicLabel
from interfaces.schemas.explain import ClassificationExplainDetail


def _sample_classification() -> ComplaintResponse:
    return ComplaintResponse(
        sentiment=PredictionDetail(
            label=SentimentLabel.NEG,
            confidence=0.9,
            explanation="neg",
        ),
        topic=PredictionDetail(
            label=TopicLabel.TECH,
            confidence=0.88,
            explanation="tech",
        ),
        intent=PredictionDetail(
            label=ActionLabel.REPORT_BUG,
            confidence=0.85,
            explanation="bug",
        ),
        action=ActionDetail(label="CREATE_JIRA_TICKET", decision_source="RULE_ENGINE"),
        meta={},
    )


app = FastAPI()
app.include_router(explain_router)
app.state.model_loader = MagicMock()
client = TestClient(app)


def test_explain_disabled_no_llm_call():
    """When LLM is disabled, classification returns without calling LLM."""
    sample = _sample_classification()

    with patch("interfaces.api.explain_route.run_pipeline", return_value=sample):
        with patch("interfaces.api.explain_route.settings") as mock_settings:
            mock_settings.LLM_ENABLED = False
            mock_settings.llm_configured = MagicMock(return_value=False)

            r = client.post("/explain-classification", json={"text": "test complaint"})
            assert r.status_code == 200
            data = r.json()
            assert data["explain_meta"]["error_code"] == "LLM_DISABLED"
            assert data["explanation"] is None
            assert data["classification"]["action"]["label"] == "CREATE_JIRA_TICKET"


def test_explain_success_with_llm():
    sample = _sample_classification()
    detail = ClassificationExplainDetail(
        summary="summary",
        rationale="rationale",
        limitations="limits",
    )

    async def fake_llm(**kwargs):
        return detail, {"explain_source": "llm", "llm_latency_ms": 12.3}, None

    with patch("interfaces.api.explain_route.run_pipeline", return_value=sample):
        with patch("interfaces.api.explain_route.settings") as mock_settings:
            mock_settings.LLM_ENABLED = True
            mock_settings.llm_configured = MagicMock(return_value=True)

            with patch(
                "interfaces.api.explain_route.llm_explain_classification",
                new=fake_llm,
            ):
                r = client.post("/explain-classification", json={"text": "test complaint"})
    assert r.status_code == 200
    data = r.json()
    assert data["explanation"]["summary"] == "summary"
    assert data["explain_meta"]["explain_source"] == "llm"


def test_explain_llm_failure_returns_classification():
    sample = _sample_classification()

    async def fake_llm(**kwargs):
        return None, {"explain_source": "fallback", "llm_latency_ms": 1.0}, "LLM_TIMEOUT"

    with patch("interfaces.api.explain_route.run_pipeline", return_value=sample):
        with patch("interfaces.api.explain_route.settings") as mock_settings:
            mock_settings.LLM_ENABLED = True
            mock_settings.llm_configured = MagicMock(return_value=True)

            with patch(
                "interfaces.api.explain_route.llm_explain_classification",
                new=fake_llm,
            ):
                r = client.post("/explain-classification", json={"text": "test complaint"})
    assert r.status_code == 200
    data = r.json()
    assert data["explanation"] is None
    assert data["explain_meta"]["error_code"] == "LLM_TIMEOUT"


def test_explain_empty_text_422():
    r = client.post("/explain-classification", json={"text": ""})
    assert r.status_code == 422
