from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from configs.config import settings
from configs.exceptions import ConfigurationError, ModelLoadError, PredictionError
from configs.logging import get_logger, setup_logging
from interfaces.api.explain_route import router as explain_router
from interfaces.api.middlewares import RequestIdMiddleware
from interfaces.api.predict_route import router as predict_router
from services.model_loader import ModelLoader

setup_logging()
logger = get_logger("app")


def error_envelope(
    error: str,
    error_code: str,
    message: str,
    details: dict | None = None,
    request: Request | None = None,
) -> dict:
    payload = {
        "error": error,
        "error_code": error_code,
        "message": message,
        "details": details or {},
    }
    if request is not None:
        rid = getattr(request.state, "request_id", None)
        if rid:
            payload["request_id"] = rid
    return payload


def _handle_startup_failure(event: str, **fields):
    """Log a startup failure. In strict mode, re-raise the current exception."""
    logger.error(event, **fields)
    if settings.ALLOW_DEGRADED_STARTUP:
        logger.warning("degraded_startup_mode_enabled")
    else:
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("text-complaint-api")

    app.state.model_loader = None
    logger.info("hf_token_check", hf_token_set=bool(settings.HF_TOKEN))
    if not settings.HF_TOKEN:
        logger.warning("hf_token_not_found", reason="HF_TOKEN not set; model loading will fail")

    logger.info("model_loading_started")
    try:
        loader = ModelLoader()
        loader.load_models()
        app.state.model_loader = loader
        logger.info("model_loading_completed")
    except ModelLoadError as e:
        _handle_startup_failure(
            "model_loading_failed",
            model_name=e.model_name,
            reason=e.reason,
            error_code=e.error_code,
        )
    except ConfigurationError as e:
        _handle_startup_failure(
            "configuration_error",
            config_key=e.config_key,
            reason=e.reason,
            error_code=e.error_code,
        )
    except Exception as e:
        _handle_startup_failure("unexpected_error_during_model_loading", reason=str(e))

    yield

    app.state.model_loader = None
    logger.info("model_unloaded")


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers (all return the unified `error_envelope` shape)
# ---------------------------------------------------------------------------

@app.exception_handler(ModelLoadError)
async def model_load_exception_handler(request: Request, exc: ModelLoadError):
    logger.error("model_load_error", exc_info=True)
    return JSONResponse(
        status_code=503,
        content=error_envelope(
            error="MODEL_LOAD_ERROR",
            error_code=exc.error_code,
            message=f"Failed to load model {exc.model_name}: {exc.reason}",
            details={"model_name": exc.model_name, "reason": exc.reason},
            request=request,
        ),
    )


@app.exception_handler(ConfigurationError)
async def config_exception_handler(request: Request, exc: ConfigurationError):
    logger.error("configuration_error", exc_info=True)
    return JSONResponse(
        status_code=400,
        content=error_envelope(
            error="CONFIGURATION_ERROR",
            error_code=exc.error_code,
            message=f"Configuration error for {exc.config_key}: {exc.reason}",
            details={"config_key": exc.config_key, "reason": exc.reason},
            request=request,
        ),
    )


@app.exception_handler(PredictionError)
async def prediction_exception_handler(request: Request, exc: PredictionError):
    logger.error("prediction_error", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=error_envelope(
            error="PREDICTION_ERROR",
            error_code=exc.error_code,
            message=f"Prediction failed: {exc.reason}",
            details={"text": exc.text[:100] if exc.text else "", "reason": exc.reason},
            request=request,
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error_code" in detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_envelope(
                error=detail.get("error", "HTTP_ERROR"),
                error_code=detail.get("error_code", "HTTP_ERROR"),
                message=detail.get("message", str(detail)),
                details=detail.get("details", {}),
                request=request,
            ),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_envelope(
            error="HTTP_ERROR",
            error_code=f"HTTP_{exc.status_code}",
            message=str(detail),
            request=request,
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=error_envelope(
            error="VALIDATION_ERROR",
            error_code="INVALID_REQUEST",
            message="Request validation failed",
            details={"errors": exc.errors()},
            request=request,
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content=error_envelope(
            error="INTERNAL_ERROR",
            error_code="UNHANDLED_EXCEPTION",
            message="An unexpected error occurred",
            request=request,
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(predict_router)
app.include_router(explain_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check(request: Request):
    loader = getattr(request.app.state, "model_loader", None)
    if loader is None or not loader.is_ready():
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "MODELS_NOT_READY",
                "message": "Models are not loaded or failed to load",
            },
        )
    return {"status": "ready"}


if settings.DEBUG_ENDPOINTS_ENABLED:
    @app.get("/debug/env")
    def debug_env():
        return {
            "HF_TOKEN": {
                "exists": settings.HF_TOKEN is not None,
                "length": len(settings.HF_TOKEN) if settings.HF_TOKEN else 0,
            },
            "thresholds": {
                "sentiment": settings.SENTIMENT_THRESHOLD,
                "topic": settings.TOPIC_THRESHOLD,
                "intent": settings.INTENT_THRESHOLD,
            },
            "flags": {
                "confidence_guarding": settings.ENABLE_CONFIDENCE_GUARDING,
                "manual_review": settings.MANUAL_REVIEW_ON_LOW_CONFIDENCE,
            },
        }
