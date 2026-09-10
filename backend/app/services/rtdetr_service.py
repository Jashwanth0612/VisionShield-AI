from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings
from app.services.drive_weights import drive_weights


RTDETR_FILENAME = "RTDETRv2_AQM_SOS_best.pth"


class RTDETRService:
    """VisionShield adapter for the project's native RT-DETRv2 AQM+SOS model."""

    def __init__(self, model_path: str | None = None, conf_threshold: float | None = None):
        configured_path = model_path or settings.rtdetr_weights_path
        self.model_path = Path(configured_path).expanduser() if configured_path else None
        self.conf_threshold = conf_threshold if conf_threshold is not None else settings.detection_confidence
        self.model: Any = None
        self.device = "cuda" if self._cuda_available() else "cpu"
        self.loaded = False
        self.load_error: str | None = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _resolve_checkpoint(self) -> Path:
        if self.model_path is not None:
            return self.model_path
        # Render's filesystem is ephemeral, so production checkpoints are
        # materialized lazily from the public Google Drive archive.
        return drive_weights.ensure(RTDETR_FILENAME)

    def load_model(self) -> None:
        self.model = None
        self.loaded = False
        self.load_error = None
        try:
            checkpoint = self._resolve_checkpoint()
            if not checkpoint.is_file():
                self.load_error = "Configured RT-DETR checkpoint was not found."
                return
            from app.services.rtdetr_native import NativeRTDETR
            self.model = NativeRTDETR(checkpoint=checkpoint, device=self.device)
            self.model.load()
            if not self.model.loaded:
                self.load_error = self.model.load_error or "Native RT-DETRv2 model is unavailable."
                return
            self.model_path = checkpoint
            self.loaded = True
        except Exception as exc:
            self.model = None
            self.load_error = f"Unable to load native RT-DETRv2 AQM+SOS checkpoint: {exc}"

    def detect(self, image: Image.Image, confidence_threshold: float | None = None) -> list[dict[str, Any]]:
        if not self.loaded or self.model is None:
            self.load_model()
        if not self.loaded or self.model is None:
            raise RuntimeError(self.load_error or "RT-DETRv2 model is unavailable.")
        threshold = confidence_threshold if confidence_threshold is not None else self.conf_threshold
        return self.model.detect(image, confidence=threshold)

    def status(self) -> dict[str, Any]:
        native_status = self.model.status() if self.model is not None else None
        return {
            "loaded": self.loaded,
            "configured": bool(self.model_path or drive_weights.is_configured(RTDETR_FILENAME)),
            "checkpoint": self.model_path.name if self.model_path else RTDETR_FILENAME,
            "checkpoint_exists": bool(self.model_path and self.model_path.is_file()),
            "confidence_threshold": self.conf_threshold,
            "device": self.device,
            "runtime": "native_rtdetrv2",
            "aqm": True,
            "sos": True,
            "error": self.load_error,
            "native": native_status,
        }
