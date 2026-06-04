class ComplaintAPIException(Exception):
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        return {
            "error": self.error_code or "UNKNOWN_ERROR",
            "message": self.message,
            "details": self.details
        }

class ModelLoadError(ComplaintAPIException):
    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        super().__init__(
            message=f"Failed to load model {model_name}: {reason}",
            error_code="MODEL_LOAD_ERROR",
            details={"model_name": model_name, "reason": reason}
        )

class PredictionError(ComplaintAPIException):
    def __init__(self, text: str, reason: str):
        self.text = text
        self.reason = reason
        super().__init__(
            message=f"Prediction failed: {reason}",
            error_code="PREDICTION_ERROR",
            details={"text": text[:100], "reason": reason}
        )

class ConfigurationError(ComplaintAPIException):
    def __init__(self, config_key: str, reason: str):
        self.config_key = config_key
        self.reason = reason
        super().__init__(
            message=f"Configuration error for {config_key}: {reason}",
            error_code="CONFIG_ERROR",
            details={"config_key": config_key, "reason": reason}
        )