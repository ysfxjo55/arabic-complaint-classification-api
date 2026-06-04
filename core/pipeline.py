from configs.config import settings
from configs.logging import get_logger
from interfaces.schemas.complaint import ActionDetail, ComplaintResponse
from interfaces.schemas.enums import ActionLabel, SentimentLabel, TopicLabel
from services.action_service import predict_action_service
from services.model_loader import ModelLoader
from services.sentiment_service import predict_sentiment_service
from services.topic_service import predict_topic_service
from utils.text_utils import ArabicInput

logger = get_logger("pipeline")


def map_action(topic: TopicLabel, sentiment: SentimentLabel, action_intent: ActionLabel) -> ActionDetail:
    source = "RULE_ENGINE"

    match (topic, sentiment, action_intent):
        case (TopicLabel.POLICY_SECURITY, _, _):
            return ActionDetail(label="BLOCK_AND_REVIEW", decision_source=source)

        case (TopicLabel.FINANCIAL, SentimentLabel.NEG, _):
            return ActionDetail(label="FINANCIAL_ESCALATION", decision_source=source)

        case (TopicLabel.TECH, _, ActionLabel.REPORT_BUG):
            return ActionDetail(label="CREATE_JIRA_TICKET", decision_source=source)

        case (TopicLabel.TECH, SentimentLabel.NEG, _):
            return ActionDetail(label="TECH_SUPPORT_ESCALATION", decision_source=source)

        case (TopicLabel.CONTENT, _, ActionLabel.USER_REQUEST):
            return ActionDetail(label="CONTENT_MODIFICATION_QUEUE", decision_source=source)

        case (_, SentimentLabel.POS, _):
            return ActionDetail(label="AUTO_REPLY_THANK_YOU", decision_source=source)

        case (_, SentimentLabel.NEU, ActionLabel.GENERAL_NOTE):
            return ActionDetail(label="ARCHIVE_NOTE", decision_source=source)

        case _:
            return ActionDetail(label="GENERAL_SUPPORT_ROUTING", decision_source=source)


def _apply_confidence_guard(
    action: ActionDetail,
    sentiment_confidence: float,
    topic_confidence: float,
    intent_confidence: float,
) -> bool:
    """Force MANUAL_REVIEW if any classifier confidence is below its threshold.
    Returns True if the guard was triggered."""
    checks = [
        ("sentiment", sentiment_confidence, settings.get_threshold("sentiment")),
        ("topic", topic_confidence, settings.get_threshold("topic")),
        ("intent", intent_confidence, settings.get_threshold("intent")),
    ]

    logger.info(
        "confidence_guard_config",
        enabled=True,
        thresholds={name: threshold for name, _, threshold in checks},
    )

    for model_name, confidence, threshold in checks:
        if confidence < threshold:
            action.label = "MANUAL_REVIEW"
            action.decision_source = "CONFIDENCE_THRESHOLD"
            logger.warning(
                "confidence_guard_triggered",
                model=model_name,
                confidence=confidence,
                threshold=threshold,
                forced_action=action.label,
            )
            return True

    return False


def run_pipeline(text: str, model_loader: ModelLoader) -> ComplaintResponse:
    raw_len = len(text or "")
    logger.info("pipeline_started", raw_text_len=raw_len)

    clean = ArabicInput(text=(text or "").strip())
    logger.info("text_cleaned", raw_text_len=raw_len, cleaned_text_len=len(clean.text))

    sentiment = predict_sentiment_service(clean.text, model_loader.sentiment_model)
    logger.info("sentiment_predicted", label=sentiment.label, confidence=sentiment.confidence)

    topic = predict_topic_service(clean.text, model_loader.topic_model)
    logger.info("topic_predicted", label=topic.label, confidence=topic.confidence)

    intent = predict_action_service(clean.text, model_loader.action_model)
    logger.info("intent_predicted", label=intent.label, confidence=intent.confidence)

    action = map_action(topic.label, sentiment.label, intent.label)
    logger.info(
        "action_mapped",
        action_label=action.label,
        decision_source=action.decision_source,
        rule_inputs={
            "topic": topic.label,
            "sentiment": sentiment.label,
            "intent": intent.label,
        },
    )

    guarding_triggered = False
    if settings.ENABLE_CONFIDENCE_GUARDING:
        guarding_triggered = _apply_confidence_guard(
            action,
            sentiment.confidence,
            topic.confidence,
            intent.confidence,
        )

    logger.info(
        "pipeline_completed",
        final_action=action.label,
        decision_source=action.decision_source,
        guarding_triggered=guarding_triggered,
    )

    return ComplaintResponse(
        sentiment=sentiment,
        topic=topic,
        intent=intent,
        action=action,
        meta={"model_version": "MARBERT-v2"},
    )
