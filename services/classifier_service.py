from enum import Enum
from typing import Mapping, TypeVar

from configs.exceptions import PredictionError
from interfaces.schemas.complaint import PredictionDetail

E = TypeVar("E", bound=Enum)


def run_classifier(
    text: str,
    model_pipeline,
    label_mapping: Mapping[str, E],
    classifier_name: str,
    low_confidence_threshold: float = 0.7,
) -> PredictionDetail:
    results = model_pipeline(text)
    if not results or not results[0]:
        raise PredictionError(
            text=text,
            reason=f"Empty model output from {classifier_name} classifier",
        )

    top = results[0][0]
    if "label" not in top or "score" not in top:
        raise PredictionError(
            text=text,
            reason=f"Malformed {classifier_name} classifier output",
        )

    label_raw = top["label"]
    if label_raw not in label_mapping:
        raise PredictionError(
            text=text,
            reason=f"Unknown {classifier_name} label from model: {label_raw}",
        )

    label_enum = label_mapping[label_raw]
    return PredictionDetail(
        label=label_enum,
        confidence=top["score"],
        explanation=f"{classifier_name.capitalize()}: {label_raw} -> {label_enum.value}",
        low_confidence=(top["score"] < low_confidence_threshold),
    )
