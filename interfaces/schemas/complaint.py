from typing import Any, Dict, Union

from pydantic import BaseModel, Field

from .enums import ActionLabel, SentimentLabel, TopicLabel


class PredictionDetail(BaseModel):
    label: Union[SentimentLabel, TopicLabel, ActionLabel]
    confidence: float
    explanation: str
    low_confidence: bool = False

class ActionDetail(BaseModel):
    label: str
    decision_source: str

class ComplaintResponse(BaseModel):
    sentiment: PredictionDetail
    topic: PredictionDetail
    intent: PredictionDetail
    action: ActionDetail
    meta: Dict[str, Any] = Field(default_factory=dict)

class ComplaintRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
