from fastapi import HTTPException, Request


def get_model_loader(request: Request):
    """Returns the loaded ModelLoader or 503 if models failed to load (e.g. missing HF_TOKEN)."""
    loader = getattr(request.app.state, "model_loader", None)
    if loader is None or not loader.is_ready():
        raise HTTPException(status_code=503, detail={
            "error_code": "MODELS_NOT_READY",
            "message": "Models are not loaded or failed to load"
        })
    else:
        return loader
