from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from interfaces.api.predict_route import router as predict_router
from interfaces.schemas.complaint import ActionDetail, ComplaintResponse, PredictionDetail
from interfaces.schemas.enums import ActionLabel, SentimentLabel, TopicLabel

app = FastAPI()
app.include_router(predict_router)

mock_loader = MagicMock()
app.state.model_loader = mock_loader

client = TestClient(app)

def test_predict_endpoint():
    mock_response = ComplaintResponse(
        sentiment=PredictionDetail(
            label=SentimentLabel.NEG,
            confidence=0.95,
            explanation="Negative sentiment detected"
        ),
        topic=PredictionDetail(
            label=TopicLabel.TECH,
            confidence=0.90,
            explanation="Related to technical issues"
        ),
        intent=PredictionDetail(
            label=ActionLabel.REPORT_BUG,
            confidence=0.85,
            explanation="User is reporting a bug"
        ),
        action=ActionDetail(
            label="CREATE_JIRA_TICKET",
            decision_source="RULE_ENGINE"
        )
    )

    with patch('interfaces.api.predict_route.run_pipeline', return_value=mock_response) as mock_pipeline:
        test_text = "التطبيق يعلق"
        
        response = client.post(
            "/predict",
            json={"text": test_text}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "sentiment" in response_data
        assert response_data["sentiment"]["label"] == SentimentLabel.NEG.value
        assert "topic" in response_data
        assert response_data["topic"]["label"] == TopicLabel.TECH.value
        assert "intent" in response_data
        assert response_data["intent"]["label"] == ActionLabel.REPORT_BUG.value
        assert "action" in response_data
        assert response_data["action"]["label"] == "CREATE_JIRA_TICKET"
        
        mock_pipeline.assert_called_once_with(test_text, mock_loader)

def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 422