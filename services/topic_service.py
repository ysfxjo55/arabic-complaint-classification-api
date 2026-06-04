from interfaces.schemas.complaint import PredictionDetail
from interfaces.schemas.enums import TopicLabel
from services.classifier_service import run_classifier

_TOPIC_MAPPING = {
    "LABEL_0": TopicLabel.POLICY_SECURITY,
    "LABEL_1": TopicLabel.FINANCIAL,
    "LABEL_2": TopicLabel.TECH,
    "LABEL_3": TopicLabel.CONTENT,
}


def predict_topic_service(text: str, model_pipeline) -> PredictionDetail:
    return run_classifier(
        text=text,
        model_pipeline=model_pipeline,
        label_mapping=_TOPIC_MAPPING,
        classifier_name="topic",
    )
