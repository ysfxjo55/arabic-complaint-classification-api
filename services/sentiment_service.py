from interfaces.schemas.complaint import PredictionDetail
from interfaces.schemas.enums import SentimentLabel
from services.classifier_service import run_classifier

_SENTIMENT_MAPPING = {
    "LABEL_0": SentimentLabel.NEG,
    "LABEL_1": SentimentLabel.NEU,
    "LABEL_2": SentimentLabel.POS,
}


def predict_sentiment_service(text: str, model_pipeline) -> PredictionDetail:
    return run_classifier(
        text=text,
        model_pipeline=model_pipeline,
        label_mapping=_SENTIMENT_MAPPING,
        classifier_name="sentiment",
    )
