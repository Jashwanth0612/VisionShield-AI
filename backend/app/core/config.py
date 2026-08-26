from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VisionShield AI API"
    version: str = "2.1.0"

    # Model weights are supplied at runtime and are never committed to the repository.
    # The NAFNet implementation uses five condition-specific checkpoints from the project:
    # Fog-ITS, Fog-OTS, Rain, Snow and Low-Light.
    nafnet_weights_path: str = ""
    nafnet_its_weights_path: str = ""
    nafnet_ots_weights_path: str = ""
    nafnet_rain_weights_path: str = ""
    nafnet_snow_weights_path: str = ""
    nafnet_lowlight_weights_path: str = ""
    rtdetr_weights_path: str = ""

    # Matches the training configuration documented in the project PPT.
    nafnet_width: int = 64
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
    def nafnet_paths(self) -> dict[str, Path]:
        configured = {
            "fog_its": self.nafnet_its_weights_path,
            "fog_ots": self.nafnet_ots_weights_path,
            "rain": self.nafnet_rain_weights_path,
            "snow": self.nafnet_snow_weights_path,
            "low_light": self.nafnet_lowlight_weights_path,
        }
        # Keep the original single-checkpoint setting as a compatibility fallback.
        if self.nafnet_weights_path and not any(configured.values()):
            configured["fog_its"] = self.nafnet_weights_path
        return {key: Path(value).expanduser() for key, value in configured.items() if value}

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
