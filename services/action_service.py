from interfaces.schemas.complaint import PredictionDetail
from interfaces.schemas.enums import ActionLabel
from services.classifier_service import run_classifier

_ACTION_MAPPING = {
    "LABEL_0": ActionLabel.GENERAL_NOTE,
    "LABEL_1": ActionLabel.USER_REQUEST,
    "LABEL_2": ActionLabel.REPORT_BUG,
}


def predict_action_service(text: str, model_pipeline) -> PredictionDetail:
    return run_classifier(
        text=text,
        model_pipeline=model_pipeline,
        label_mapping=_ACTION_MAPPING,
        classifier_name="action",
    )
