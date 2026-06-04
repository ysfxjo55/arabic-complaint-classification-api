import time
import uuid

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from structlog.contextvars import bind_contextvars

logger = structlog.get_logger("http")

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)
        start_time = time.time()
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000
            logger.info("Request completed",status_code=response.status_code, duration_ms=round(duration_ms, 2))

            response.headers["x-request-id"] = request_id
            return response
        except Exception:
            duration_ms = (time.time() - start_time) * 1000
            logger.exception("Request failed", duration_ms=round(duration_ms, 2))
            raise
