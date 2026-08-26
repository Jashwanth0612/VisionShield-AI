from __future__ import annotations

import time
import uuid
from typing import Any

from PIL import Image

from app.api.pipeline import nafnet_service, rtdetr_service
from app.services.storage import storage


def benchmark_image(image: Image.Image, filename: str, runs: int = 3, enhancement: bool = True, confidence: float | None = None) -> dict[str, Any]:
    """Measure end-to-end image inference and persist one explicit benchmark action."""
    if enhancement and not nafnet_service.loaded:
        raise RuntimeError(nafnet_service.load_error or "NAFNet is unavailable.")
    if not rtdetr_service.loaded:
        raise RuntimeError(rtdetr_service.load_error or "RT-DETR is unavailable.")

    runs = max(1, min(int(runs), 10))
    latencies: list[float] = []
    detection_counts: list[int] = []
    for _ in range(runs):
        start = time.perf_counter()
        processed = nafnet_service.enhance_image(image) if enhancement else image.copy()
        detections = rtdetr_service.detect(processed, confidence_threshold=confidence)
        latencies.append((time.perf_counter() - start) * 1000)
        detection_counts.append(len(detections))

    avg_ms = sum(latencies) / len(latencies)
    run_id = f"bench_{uuid.uuid4().hex[:12]}"
    timestamp = storage.now()
    model_config = f"NAFNet={'on' if enhancement else 'off'} · RT-DETR conf={confidence if confidence is not None else rtdetr_service.conf_threshold:.2f}"
    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "media_type": "image",
        "filename": filename or "benchmark-image",
        "iterations": runs,
        "latency_ms": round(avg_ms, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "fps": round(1000 / avg_ms, 2) if avg_ms else 0,
        "detections": round(sum(detection_counts) / len(detection_counts), 2),
        "model_config": model_config,
        "details": {"raw_latencies_ms": [round(value, 2) for value in latencies], "detection_counts": detection_counts, "models": {"nafnet": nafnet_service.status(), "rt_detr": rtdetr_service.status()}},
    }
    storage.record_benchmark(record)
    return record
