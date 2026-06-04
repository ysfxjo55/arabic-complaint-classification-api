from typing import Optional

import torch
from transformers import pipeline

from configs.config import settings
from configs.exceptions import ConfigurationError, ModelLoadError
from configs.logging import get_logger

logger = get_logger("model_loader")

device_id = 0 if torch.cuda.is_available() else -1
logger.info("device_selected", device_id=device_id, cuda=torch.cuda.is_available())


class ModelLoader:
    def __init__(self) -> None:
        self.sentiment_model: Optional[object] = None
        self.topic_model: Optional[object] = None
        self.action_model: Optional[object] = None
        self.device: int = device_id

    def _load_one(self, name: str, repo: str, hf_token: str):
        try:
            logger.info("loading_model", name=name, repo=repo)
            model = pipeline(
                "text-classification",
                model=repo,
                device=self.device,
                top_k=3,
                token=hf_token,
            )
            logger.info("model_loaded", name=name)
            return model
        except Exception as e:
            logger.error("model_load_failed", name=name, reason=str(e))
            raise ModelLoadError(name, str(e))

    def load_models(self) -> None:
        hf_token = settings.HF_TOKEN
        if not hf_token:
            raise ConfigurationError("HF_TOKEN", "Environment variable not found")

        self.sentiment_model = self._load_one("sentiment", settings.HF_MODEL_SENTIMENT, hf_token)
        self.topic_model = self._load_one("topic", settings.HF_MODEL_TOPIC, hf_token)
        self.action_model = self._load_one("action", settings.HF_MODEL_ACTION, hf_token)

        logger.info("all_models_loaded")

    def is_ready(self) -> bool:
        return (
            self.sentiment_model is not None
            and self.topic_model is not None
            and self.action_model is not None
        )
