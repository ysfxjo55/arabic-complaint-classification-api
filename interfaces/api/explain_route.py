from fastapi import APIRouter, Depends

from configs.config import settings
from configs.logging import get_logger
from core.pipeline import run_pipeline
from interfaces.api.dependencies import get_model_loader
from interfaces.schemas.explain import ExplainClassificationRequest, ExplainClassificationResponse
from services.llm_service import explain_classification as llm_explain_classification

router = APIRouter(tags=["Explanation"])
logger = get_logger("explain_route")


@router.post("/explain-classification", response_model=ExplainClassificationResponse)
async def explain_classification_endpoint(
    body: ExplainClassificationRequest,
    loader=Depends(get_model_loader),
):
    """
    Runs the full deterministic pipeline first, then optionally calls the LLM to explain
    the fixed labels and action (LLM does not decide routing).
    """
    classification = run_pipeline(body.text, loader)

    explain_meta: dict = {}

    if not settings.LLM_ENABLED or not settings.llm_configured():
        return ExplainClassificationResponse(
            classification=classification,
            explanation=None,
            explain_meta={
                "explain_source": "disabled",
                "error_code": "LLM_DISABLED" if not settings.LLM_ENABLED else "MISSING_API_KEY",
                "llm_latency_ms": None,
            },
        )

    detail, meta, err = await llm_explain_classification(
        text=body.text,
        classification=classification,
        settings=settings,
    )
    explain_meta = {**meta}
    if err:
        explain_meta["error_code"] = err

    return ExplainClassificationResponse(
        classification=classification,
        explanation=detail,
        explain_meta=explain_meta,
    )
