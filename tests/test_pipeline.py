from unittest.mock import MagicMock, patch

from core.pipeline import (
    map_action,
    run_pipeline,
)
from interfaces.schemas.complaint import (
    ComplaintResponse,
    PredictionDetail,
)
from interfaces.schemas.enums import ActionLabel, SentimentLabel, TopicLabel
from services.action_service import predict_action_service
from services.sentiment_service import predict_sentiment_service
from services.topic_service import predict_topic_service


class MockModelLoader:
    """Mock model loader exposing the same `.sentiment_model`, `.topic_model`,
    `.action_model` attributes as the real one, each returning the HF
    text-classification pipeline shape `[[{"label": ..., "score": ...}, ...]]`.
    """

    def __init__(self):
        self.sentiment_model = MagicMock(
            return_value=[[{"label": "LABEL_0", "score": 0.95}]]
        )
        self.topic_model = MagicMock(
            return_value=[[{"label": "LABEL_2", "score": 0.90}]]
        )
        self.action_model = MagicMock(
            return_value=[[{"label": "LABEL_1", "score": 0.85}]]
        )
        self.device = -1


class TestPredictSentimentService:
    def test_returns_prediction_detail(self):
        loader = MockModelLoader()
        result = predict_sentiment_service("This is a test", loader.sentiment_model)
        assert isinstance(result, PredictionDetail)
        assert result.label == SentimentLabel.NEG
        assert result.confidence == 0.95


class TestPredictTopicService:
    def test_returns_prediction_detail(self):
        loader = MockModelLoader()
        result = predict_topic_service("This is a test", loader.topic_model)
        assert isinstance(result, PredictionDetail)
        # LABEL_2 maps to TopicLabel.TECH in the topic mapping
        assert result.label == TopicLabel.TECH
        assert result.confidence == 0.90


class TestPredictActionService:
    def test_returns_prediction_detail(self):
        loader = MockModelLoader()
        result = predict_action_service("This is a test", loader.action_model)
        assert isinstance(result, PredictionDetail)
        assert result.label == ActionLabel.USER_REQUEST
        assert result.confidence == 0.85


class TestMapAction:
    def test_map_action_security_topic_blocks(self):
        result = map_action(
            sentiment=SentimentLabel.POS,
            topic=TopicLabel.POLICY_SECURITY,
            action_intent=ActionLabel.USER_REQUEST,
        )
        assert result.label == "BLOCK_AND_REVIEW"
        assert result.decision_source == "RULE_ENGINE"

    def test_map_action_negative_financial_escalates(self):
        result = map_action(
            sentiment=SentimentLabel.NEG,
            topic=TopicLabel.FINANCIAL,
            action_intent=ActionLabel.USER_REQUEST,
        )
        assert result.label == "FINANCIAL_ESCALATION"

    def test_map_action_tech_bug_creates_ticket(self):
        result = map_action(
            sentiment=SentimentLabel.NEU,
            topic=TopicLabel.TECH,
            action_intent=ActionLabel.REPORT_BUG,
        )
        assert result.label == "CREATE_JIRA_TICKET"

    def test_map_action_negative_tech_escalates(self):
        result = map_action(
            sentiment=SentimentLabel.NEG,
            topic=TopicLabel.TECH,
            action_intent=ActionLabel.USER_REQUEST,
        )
        assert result.label == "TECH_SUPPORT_ESCALATION"

    def test_map_action_content_modification(self):
        result = map_action(
            sentiment=SentimentLabel.NEU,
            topic=TopicLabel.CONTENT,
            action_intent=ActionLabel.USER_REQUEST,
        )
        assert result.label == "CONTENT_MODIFICATION_QUEUE"

    def test_map_action_positive_sentiment_thanks(self):
        result = map_action(
            sentiment=SentimentLabel.POS,
            topic=TopicLabel.CONTENT,
            action_intent=ActionLabel.GENERAL_NOTE,
        )
        assert result.label == "AUTO_REPLY_THANK_YOU"

    def test_map_action_neutral_note_archives(self):
        result = map_action(
            sentiment=SentimentLabel.NEU,
            topic=TopicLabel.CONTENT,
            action_intent=ActionLabel.GENERAL_NOTE,
        )
        assert result.label == "ARCHIVE_NOTE"

    def test_map_action_default_routing(self):
        result = map_action(
            sentiment=SentimentLabel.NEU,
            topic=TopicLabel.FINANCIAL,
            action_intent=ActionLabel.USER_REQUEST,
        )
        assert result.label == "GENERAL_SUPPORT_ROUTING"


class TestRunPipeline:
    @patch("core.pipeline.predict_action_service")
    @patch("core.pipeline.predict_topic_service")
    @patch("core.pipeline.predict_sentiment_service")
    def test_run_pipeline_returns_complaint_response(
        self, mock_sentiment, mock_topic, mock_intent
    ):
        loader = MockModelLoader()
        mock_sentiment.return_value = PredictionDetail(
            label=SentimentLabel.NEG, confidence=0.95, explanation="Test sentiment"
        )
        mock_topic.return_value = PredictionDetail(
            label=TopicLabel.TECH, confidence=0.9, explanation="Test topic"
        )
        mock_intent.return_value = PredictionDetail(
            label=ActionLabel.USER_REQUEST, confidence=0.85, explanation="Test intent"
        )

        result = run_pipeline("This is a test complaint", loader)

        assert isinstance(result, ComplaintResponse)
        assert result.sentiment.label == SentimentLabel.NEG
        assert result.topic.label == TopicLabel.TECH
        assert result.intent.label == ActionLabel.USER_REQUEST
        # (TECH, NEG, USER_REQUEST) -> TECH_SUPPORT_ESCALATION
        assert result.action.label == "TECH_SUPPORT_ESCALATION"

    @patch("core.pipeline.ArabicInput")
    @patch("core.pipeline.predict_action_service")
    @patch("core.pipeline.predict_topic_service")
    @patch("core.pipeline.predict_sentiment_service")
    def test_run_pipeline_cleans_input_text(
        self, mock_sentiment, mock_topic, mock_intent, mock_arabic_input
    ):
        mock_clean_text = MagicMock()
        mock_clean_text.text = "cleaned text"
        mock_arabic_input.return_value = mock_clean_text

        loader = MockModelLoader()

        mock_sentiment.return_value = PredictionDetail(
            label=SentimentLabel.NEU, confidence=0.5, explanation="Test"
        )
        mock_topic.return_value = PredictionDetail(
            label=TopicLabel.CONTENT, confidence=0.5, explanation="Test"
        )
        mock_intent.return_value = PredictionDetail(
            label=ActionLabel.USER_REQUEST, confidence=0.5, explanation="Test"
        )

        run_pipeline("  dirty text with spaces  ", loader)

        mock_arabic_input.assert_called_once_with(text="dirty text with spaces")
        mock_sentiment.assert_called_once_with("cleaned text", loader.sentiment_model)
        mock_topic.assert_called_once_with("cleaned text", loader.topic_model)
        mock_intent.assert_called_once_with("cleaned text", loader.action_model)
