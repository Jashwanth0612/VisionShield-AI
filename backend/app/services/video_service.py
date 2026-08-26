from __future__ import annotations

import io
import os
import tempfile
import time
import uuid
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.api.pipeline import nafnet_service, rtdetr_service
from app.core.config import settings
from app.services.storage import storage

MAX_VIDEO_BYTES = settings.max_video_mb * 1024 * 1024
MAX_VIDEO_SECONDS = settings.max_video_seconds


def _annotate_frame(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    frame = image.copy()
    draw = ImageDraw.Draw(frame)
    width = max(2, min(6, image.width // 400))
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        label = f'{item["label"]} {item["confidence"]:.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=(0, 240, 255), width=width)
        draw.text((x1 + 3, y1 + 3), label, fill=(248, 250, 252))
    return frame


def analyze_video(contents: bytes, filename: str, sample_fps: float = 2.0, enhancement: bool = True, confidence: float | None = None) -> dict[str, Any]:
    """Run sampled NAFNet + RT-DETR inference and write a sampled annotated result video."""
    if enhancement and not nafnet_service.loaded:
        raise RuntimeError(nafnet_service.load_error or "NAFNet is unavailable.")
    if not rtdetr_service.loaded:
        raise RuntimeError(rtdetr_service.load_error or "RT-DETR is unavailable.")
    if len(contents) > MAX_VIDEO_BYTES:
        raise ValueError(f"Video exceeds the {settings.max_video_mb} MB upload limit.")
    sample_fps = max(0.5, min(float(sample_fps), 10.0))

    tmp_path: str | None = None
    annotated_path: str | None = None
    enhanced_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=PathSuffix(filename), delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        capture = cv2.VideoCapture(tmp_path)
        if not capture.isOpened():
            raise ValueError("Unable to decode the uploaded video.")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / source_fps if source_fps > 0 else 0
        if duration > MAX_VIDEO_SECONDS:
            raise ValueError(f"Video exceeds the {MAX_VIDEO_SECONDS} second analysis limit.")

        stride = max(1, round(source_fps / sample_fps)) if source_fps > 0 else 1
        processed_frames = 0
        detections_total = 0
        labels: dict[str, int] = {}
        total_inference_ms = 0.0
        started = time.perf_counter()
        annotated_writer = None
        enhanced_writer = None
        frame_size = None

        annotated_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        annotated_path = annotated_temp.name
        annotated_temp.close()
        enhanced_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        enhanced_path = enhanced_temp.name
        enhanced_temp.close()

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
            processed = nafnet_service.enhance_image(image) if enhancement else image.copy()
            detections = rtdetr_service.detect(processed, confidence_threshold=confidence)
            total_inference_ms += (time.perf_counter() - inference_start) * 1000

            annotated = _annotate_frame(processed, detections)
            if frame_size is None:
                frame_size = (processed.width, processed.height)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                annotated_writer = cv2.VideoWriter(annotated_path, fourcc, sample_fps, frame_size)
                if enhancement:
                    enhanced_writer = cv2.VideoWriter(enhanced_path, fourcc, sample_fps, frame_size)
            annotated_writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
            if enhanced_writer:
                enhanced_writer.write(cv2.cvtColor(np.asarray(processed), cv2.COLOR_RGB2BGR))

            processed_frames += 1
            detections_total += len(detections)
            for item in detections:
                label = str(item.get("label", "unknown"))
                labels[label] = labels.get(label, 0) + 1
            frame_index += 1

        capture.release()
        if annotated_writer:
            annotated_writer.release()
        if enhanced_writer:
            enhanced_writer.release()

        if processed_frames == 0:
            raise ValueError("No decodable frames were available at the selected sampling rate.")

        elapsed_ms = (time.perf_counter() - started) * 1000
        run_id = f"vid_{uuid.uuid4().hex[:12]}"
        original_artifact = storage.save_artifact(contents, run_id, "original", filename, "video/mp4")
        annotated_artifact = storage.save_artifact(Path(annotated_path).read_bytes(), run_id, "annotated", "annotated-sampled.mp4", "video/mp4")
        enhanced_artifact = storage.save_artifact(Path(enhanced_path).read_bytes(), run_id, "enhanced", "enhanced-sampled.mp4", "video/mp4") if enhancement else None

        latency = round(total_inference_ms / processed_frames, 2)
        analysis_fps = round(processed_frames / (elapsed_ms / 1000), 2) if elapsed_ms else 0
        models = {"nafnet": nafnet_service.status(), "rt_detr": rtdetr_service.status()}
        timestamp = storage.now()
        storage.record_inference({
            "run_id": run_id,
            "timestamp": timestamp,
            "media_type": "video",
            "filename": filename or "video",
            "nafnet_enabled": enhancement,
            "confidence_threshold": confidence if confidence is not None else rtdetr_service.conf_threshold,
            "detections": detections_total,
            "latency_ms": latency,
            "fps": analysis_fps,
            "model_status": "ready",
            "model_config": f"NAFNet={'on' if enhancement else 'off'} · RT-DETR conf={confidence if confidence is not None else rtdetr_service.conf_threshold:.2f}",
            "artifacts": {"original": original_artifact, "enhanced": enhanced_artifact, "annotated": annotated_artifact},
            "details": {"duration_seconds": round(duration, 2), "source_fps": round(source_fps, 2), "sample_fps": sample_fps, "frames_analyzed": processed_frames, "detections_per_frame": round(detections_total / processed_frames, 2), "detected_classes": labels, "models": models},
        })

        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "duration_seconds": round(duration, 2),
            "source_fps": round(source_fps, 2),
            "sample_fps": sample_fps,
            "frames_analyzed": processed_frames,
            "detections_total": detections_total,
            "detections_per_frame": round(detections_total / processed_frames, 2),
            "inference_latency_ms": latency,
            "analysis_fps": analysis_fps,
            "detected_classes": dict(sorted(labels.items(), key=lambda item: (-item[1], item[0]))),
            "enhancement_enabled": enhancement,
            "models": models,
            "artifacts": {"original": original_artifact, "enhanced": enhanced_artifact, "annotated": annotated_artifact},
        }
    finally:
        for path in (tmp_path, annotated_path, enhanced_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def PathSuffix(filename: str | None) -> str:
    suffix = os.path.splitext(filename or "video.mp4")[1].lower()
    return suffix if suffix in {".mp4", ".mov", ".avi", ".webm", ".mpeg", ".mpg"} else ".mp4"
