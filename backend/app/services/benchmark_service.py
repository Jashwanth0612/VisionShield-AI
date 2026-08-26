from __future__ import annotations

import time
from typing import Any

from PIL import Image

from app.api.pipeline import nafnet_service, rtdetr_service


def benchmark_image(image: Image.Image, runs: int = 3, enhancement: bool = True) -> dict[str, Any]:
    """Measure end-to-end inference without claiming accuracy metrics."""
    runs = max(1, min(int(runs), 10))
    latencies: list[float] = []
    detection_counts: list[int] = []

    for _ in range(runs):
        start = time.perf_counter()
        processed = nafnet_service.enhance_image(image) if enhancement else image.copy()
        detections = rtdetr_service.detect(processed)
        latencies.append((time.perf_counter() - start) * 1000)
        detection_counts.append(len(detections))

    avg_ms = sum(latencies) / len(latencies)
    return {
        "runs": runs,
        "enhancement_enabled": enhancement,
        "latency_ms": {
            "average": round(avg_ms, 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "fps_equivalent": round(1000 / avg_ms, 2) if avg_ms else 0,
        "average_detections": round(sum(detection_counts) / len(detection_counts), 2),
        "models": {
            "nafnet": nafnet_service.status(),
            "rt_detr": rtdetr_service.status(),
        },
    }
