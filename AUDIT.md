# System Audit — April 2026

Snapshot of failure surfaces, integration points, and untested assumptions across
the text-complaint-api and its agent integrations. Written as part of the April
"system audit & failure surfaces" milestone.

---

## 1. Integration points

| Integration | Direction | Surface |
|-------------|-----------|---------|
| Voiceflow | Inbound | `POST /predict`, `POST /explain-classification` |
| github_cleaner agent | Side system | Calls API for classification of issue/PR text |
| Hugging Face Hub | Outbound | Model downloads on startup (sentiment, topic, action) |
| OpenAI-compatible LLM (`LLM_BASE_URL`) | Outbound | `/chat/completions` for explanation only |
| MLflow | Side store | Tracing for LLM call + experiment metadata |

The API is the **deterministic source of truth**. LLM and logging are
side-effects and never decide routing.

---

## 2. Failure modes

### Startup
- Missing or invalid `HF_TOKEN` → `ConfigurationError`, app fails to start (strict mode).
- Hugging Face repo 404 / network error → `ModelLoadError`, app fails to start.
- MLflow tracking URI not writable (e.g., local default `file:///app/mlruns`) → startup crashes
  before models load. Mitigated by documenting `MLFLOW_TRACKING_URI=file:./mlruns` for local.
- Degraded mode (`ALLOW_DEGRADED_STARTUP=true`) allows app to start even if models fail; `/ready`
  still returns 503 so traffic gating is preserved.

### Request path
- Empty / oversize input → 422 `INVALID_REQUEST` (Pydantic).
- Models not loaded → 503 `MODELS_NOT_READY` (via dependency guard).
- Classifier returns unknown label or malformed shape → 500 `PREDICTION_ERROR`
  (no silent default).
- Unhandled exception anywhere → 500 `UNHANDLED_EXCEPTION` envelope.
- Any HTTP exception (incl. 404, 405) → wrapped into unified envelope.

### LLM path (`/explain-classification`)
- LLM disabled → `LLM_DISABLED`, deterministic classification still returned.
- Missing API key → `MISSING_API_KEY`.
- Timeout → `LLM_TIMEOUT`.
- Non-2xx or network failure → `LLM_HTTP_ERROR`.
- Malformed JSON or schema mismatch → `LLM_INVALID_RESPONSE`.
- In all LLM failures: HTTP 200, `explanation: null`, `explain_meta.error_code` set.

### Side effects
- `save_prediction_log` is wrapped in try/except; failure logs a warning, never fails the request.

---

## 3. Observability

- Every request gets a `request_id` (incoming `x-request-id` header reused if present).
- `request_id` flows through:
  - `structlog` contextvars (every log line)
  - `request.state.request_id` (used by error envelope)
  - response body (`request_id` field on all error responses)
  - response header (`x-request-id`)
- Stage-level logs per request:
  `pipeline_started → text_cleaned → sentiment_predicted → topic_predicted →
  intent_predicted → action_mapped → confidence_guard_config → pipeline_completed →
  Request completed`.
- LLM call is traced via `mlflow.trace` + `mlflow.start_span("httpx_chat_completions")`.
- No raw user text is logged (PII gap closed during this audit).

---

## 4. Untested assumptions (known unknowns)

- Hugging Face transformers `pipeline(...)` output is always `[[{"label","score"}, ...]]`.
  We added shape guards, but we have not stress-tested with adversarial / unusual model outputs.
- `predictions.json` file logging is single-writer; under multi-worker deployment it could
  produce inconsistent files (not currently used in production).
- MLflow tracing assumes the tracking URI is reachable and writable; we degrade silently on
  network failures (acceptable, since tracing is side-effect).
- Confidence thresholds (`SENTIMENT_THRESHOLD`, etc.) are static; no automatic recalibration.
- We have no integration test that simulates LLM provider returning HTTP 500.
- We have no integration test for `ALLOW_DEGRADED_STARTUP=true` path.
- `/debug/env` is gated behind `DEBUG_ENDPOINTS_ENABLED` (default false) — disabled in production.

---

## 5. What we have not tested at all

- Sustained load (>10 RPS on `/predict`).
- Concurrency on `save_prediction_log` under multiple workers.
- Behavior with extremely long or adversarial Arabic text inputs.
- Behavior when Hugging Face Hub is reachable but slow (only fast/timeout cases tested).
- Voiceflow timeout behavior on `/explain-classification`.

---

## 6. Closed during this audit

- Silent label defaults in sentiment/topic/action services → now reject unknown labels.
- Mixed `print` + structured logging → unified on `structlog`.
- `/health` returning OK regardless of model state → `/ready` added with real readiness check.
- Inconsistent error envelopes → unified via `error_envelope` helper + global handlers.
- Raw user text in logs → removed.
- `save_prediction_log` could break `/predict` → wrapped, hardened against corrupt JSON.
- `model_loader.is_ready` syntax error → fixed (would prevent any startup).
- `.env.example` missing most variables → fully expanded with safe placeholders.

---

## 7. Top remaining risks

1. **CI is informational** — `ruff format` and UI lint use `|| true`; regressions can still slip through.
2. **No failure-drill tests.** Cold start, bad LLM JSON, missing key, upstream timeout
   are validated by reading code, not by running tests.
3. **No retries / circuit breaker** for the LLM HTTP call.
4. **No request-level timeout** around the deterministic pipeline.
5. **`/debug/env`** is gated behind `DEBUG_ENDPOINTS_ENABLED` (default false; off in production).
