from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.api.pipeline import nafnet_service, rtdetr_service

MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_SECONDS = 120


def analyze_video(contents: bytes, sample_fps: float = 2.0, enhancement: bool = True, confidence: float | None = None) -> dict[str, Any]:
    """Run sampled-frame analysis and return real-time CV metrics.

    This intentionally returns metrics rather than a generated video artifact, keeping the
    API stateless and suitable for a first deployment. The same NAFNet -> RT-DETR pipeline
    used by image inference is applied to sampled frames.
    """
    if len(contents) > MAX_VIDEO_BYTES:
        raise ValueError("Video exceeds the 100 MB upload limit.")
    sample_fps = max(0.5, min(float(sample_fps), 10.0))

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        capture = cv2.VideoCapture(tmp_path)
        if not capture.isOpened():
            raise ValueError("Unable to decode the uploaded video.")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0
        if duration > MAX_VIDEO_SECONDS:
            raise ValueError("Video exceeds the 120 second analysis limit.")

        stride = max(1, round(fps / sample_fps)) if fps > 0 else 1
        processed_frames = 0
        total_inference_ms = 0.0
        detections_total = 0
        labels: dict[str, int] = {}
        started = time.perf_counter()

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            inference_start = time.perf_counter()
            processed = nafnet_service.enhance_image(image) if enhancement else image
            detections = rtdetr_service.detect(processed, confidence_threshold=confidence)
            total_inference_ms += (time.perf_counter() - inference_start) * 1000
            processed_frames += 1
            detections_total += len(detections)
            for item in detections:
                label = str(item.get("label", "unknown"))
                labels[label] = labels.get(label, 0) + 1
            frame_index += 1

        capture.release()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "duration_seconds": round(duration, 2),
            "source_fps": round(fps, 2),
            "sample_fps": sample_fps,
            "frames_analyzed": processed_frames,
            "detections_total": detections_total,
            "detections_per_frame": round(detections_total / processed_frames, 2) if processed_frames else 0,
            "inference_latency_ms": round(total_inference_ms / processed_frames, 2) if processed_frames else 0,
            "analysis_fps": round(processed_frames / (elapsed_ms / 1000), 2) if elapsed_ms else 0,
            "detected_classes": dict(sorted(labels.items(), key=lambda item: (-item[1], item[0]))),
            "enhancement_enabled": enhancement,
            "models": {"nafnet": nafnet_service.status(), "rt_detr": rtdetr_service.status()},
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
