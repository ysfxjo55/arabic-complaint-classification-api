import json
import os
from datetime import datetime

from fastapi import APIRouter, Depends

from configs.config import settings
from configs.logging import get_logger
from core.pipeline import run_pipeline
from interfaces.api.dependencies import get_model_loader
from interfaces.schemas.complaint import ComplaintRequest, ComplaintResponse

logger = get_logger("predict_route")

router = APIRouter(prefix="/predict", tags=["Prediction"])

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
LOGS_FILE = os.path.join(LOGS_DIR, "predictions.json")

def save_prediction_log(input_text: str, response: ComplaintResponse):
    os.makedirs(LOGS_DIR, exist_ok=True)

    entries = []
    if os.path.exists(LOGS_FILE):
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                entries = json.load(f)
                if not isinstance(entries, list):
                    entries = []
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append({
        "timestamp": datetime.now().isoformat(),
        "input_text": input_text,
        "response": response.model_dump(mode="json"),
    })

    with open(LOGS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

@router.post("", response_model=ComplaintResponse)
async def predict_complaint(request: ComplaintRequest, loader = Depends(get_model_loader)):
    result = run_pipeline(request.text, loader)
    if settings.ENABLE_PREDICTION_LOGGING:
        try:
            save_prediction_log(request.text, result)
        except Exception as exc:
            logger.warning(
                "prediction_log_failed",
                error=str(exc),
            )
    return result
