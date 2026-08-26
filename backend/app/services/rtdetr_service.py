from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings


class RTDETRService:
    """RT-DETR inference adapter with configurable local weights and safe fallback."""

    def __init__(self, model_path: str | None = None, conf_threshold: float | None = None):
        self.model_path = Path(model_path or settings.rtdetr_model)
        self.conf_threshold = conf_threshold or settings.detection_confidence
        self.model: Any = None
        self.device = "cpu"
        self.loaded = False
        self.load_error: str | None = None

    def load_model(self) -> None:
        try:
            from ultralytics import RTDETR

            checkpoint = str(self.model_path) if self.model_path.exists() else "rtdetr-l.pt"
            self.model = RTDETR(checkpoint)
            self.loaded = True
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.loaded = False
            self.load_error = str(exc)

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.model is None:
            return []

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.conf_threshold
        )
        results = self.model.predict(source=image, conf=threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        names = result.names or {}
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
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
            "checkpoint": str(self.model_path),
            "checkpoint_exists": self.model_path.exists(),
            "confidence_threshold": self.conf_threshold,
            "error": self.load_error,
        }
