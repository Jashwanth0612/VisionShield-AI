from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings


class RTDETRService:
    """RT-DETR inference adapter using only explicitly configured local weights."""

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

    def load_model(self) -> None:
        self.model = None
        self.loaded = False
        self.load_error = None
        if self.model_path is None:
            self.load_error = "RT-DETR weights are not configured. Set RTDETR_WEIGHTS_PATH."
            return
        if not self.model_path.is_file():
            self.load_error = "Configured RT-DETR weight file was not found."
            return
        try:
            from ultralytics import RTDETR
            self.model = RTDETR(str(self.model_path))
            self.loaded = True
        except Exception as exc:
            self.model = None
            self.load_error = f"Unable to load RT-DETR checkpoint: {exc}"

    def detect(self, image: Image.Image, confidence_threshold: float | None = None) -> list[dict[str, Any]]:
        if not self.loaded or self.model is None:
            raise RuntimeError(self.load_error or "RT-DETR model is unavailable.")
        threshold = confidence_threshold if confidence_threshold is not None else self.conf_threshold
        results = self.model.predict(source=image, conf=threshold, verbose=False, device=self.device)
        if not results:
            return []
        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None:
            return []
        detections: list[dict[str, Any]] = []
        for box in boxes:
            xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
            confidence = round(float(box.conf[0]), 4)
            class_id = int(box.cls[0])
            detections.append({
                "bbox": xyxy,
                "confidence": confidence,
                "class_id": class_id,
                "label": names.get(class_id, str(class_id)),
            })
        return detections

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "configured": bool(self.model_path),
            "checkpoint": self.model_path.name if self.model_path else None,
            "checkpoint_exists": bool(self.model_path and self.model_path.is_file()),
            "confidence_threshold": self.conf_threshold,
            "device": self.device,
            "error": self.load_error,
        }
