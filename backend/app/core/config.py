from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VisionShield AI API"
    version: str = "1.0.0"
    model_dir: str = "models"
    rtdetr_model: str = "models/rt_detr/rtdetr_weights.pt"
    nafnet_checkpoint: str = "models/nafnet/nafnet_weights.pth"
    detection_confidence: float = 0.35
    max_upload_mb: int = 20
    cors_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def model_directory(self) -> Path:
        return Path(self.model_dir)

    @property
    def rtdetr_path(self) -> Path:
        return Path(self.rtdetr_model)

    @property
    def nafnet_path(self) -> Path:
        return Path(self.nafnet_checkpoint)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
