from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VisionShield AI API"
    version: str = "2.0.0"

    # Never commit model weights. Supply absolute or container-mounted paths at runtime.
    nafnet_weights_path: str = ""
    rtdetr_weights_path: str = ""

    # NAFNet architecture knobs match the official NAFNet family; tune them to the supplied checkpoint.
    nafnet_width: int = 32
    nafnet_middle_blocks: int = 1
    nafnet_encoder_blocks: str = "1,1,1,28"
    nafnet_decoder_blocks: str = "1,1,1,1"

    detection_confidence: float = 0.35
    max_upload_mb: int = 20
    max_video_mb: int = 100
    max_video_seconds: int = 120
    cors_origins: str = "*"

    artifact_dir: str = "data/artifacts"
    database_path: str = "data/visionshield.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def nafnet_path(self) -> Path:
        return Path(self.nafnet_weights_path).expanduser() if self.nafnet_weights_path else Path()

    @property
    def rtdetr_path(self) -> Path:
        return Path(self.rtdetr_weights_path).expanduser() if self.rtdetr_weights_path else Path()

    @property
    def encoder_blocks(self) -> list[int]:
        return [int(value.strip()) for value in self.nafnet_encoder_blocks.split(",") if value.strip()]

    @property
    def decoder_blocks(self) -> list[int]:
        return [int(value.strip()) for value in self.nafnet_decoder_blocks.split(",") if value.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
