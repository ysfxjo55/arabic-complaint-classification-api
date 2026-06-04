from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from interfaces.schemas.complaint import ComplaintResponse


class ClassificationExplainDetail(BaseModel):
    """Structured LLM output: explains deterministic classification only."""

    summary: str = Field(..., description="Brief summary of the complaint and classification outcome")
    rationale: str = Field(
        ...,
        description="Why the rule-based action follows from sentiment, topic, and intent",
    )
    limitations: str = Field(
        default="",
        description="Clarifies that routing decisions are deterministic, not LLM-decided",
    )


class ExplainClassificationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ExplainClassificationResponse(BaseModel):
    classification: ComplaintResponse
    explanation: Optional[ClassificationExplainDetail] = None
    explain_meta: Dict[str, Any] = Field(default_factory=dict)
