from __future__ import annotations

import base64
import io
import time
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from app.services.nafnet_service import NAFNetService
from app.services.rtdetr_service import RTDETRService

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])
nafnet_service = NAFNetService()
rtdetr_service = RTDETRService()


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _annotate(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = f'{detection["label"]} {detection["confidence"]:.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 120), width=max(2, image.width // 400))
        try:
            bbox = draw.textbbox((x1, y1), label)
            draw.rectangle(bbox, fill=(0, 0, 0))
        except Exception:
            pass
        draw.text((x1 + 3, y1 + 2), label, fill=(255, 255, 255))
    return annotated


def _validate_image(file: UploadFile, contents: bytes) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image.") from exc


@router.post("/process")
async def process_image(
    file: UploadFile = File(...),
    enable_enhancement: bool = Query(True),
    confidence: float | None = Query(None, ge=0.05, le=0.99),
):
    """Enhance an image with NAFNet and detect objects with RT-DETR."""
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB upload limit.")
    image = _validate_image(file, contents)

    if confidence is not None:
        rtdetr_service.conf_threshold = confidence

    pipeline_start = time.perf_counter()
    enhancement_start = time.perf_counter()
    processed = nafnet_service.enhance_image(image) if enable_enhancement else image.copy()
    enhancement_ms = round((time.perf_counter() - enhancement_start) * 1000, 2)

    detection_start = time.perf_counter()
    detections = rtdetr_service.detect(processed)
    detection_ms = round((time.perf_counter() - detection_start) * 1000, 2)
    annotated = _annotate(processed, detections)
    total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

    return {
        "status": "success",
        "filename": file.filename,
        "image_size": {"width": image.width, "height": image.height},
        "enhancement_enabled": enable_enhancement,
        "models": {
            "nafnet": nafnet_service.status(),
            "rt_detr": rtdetr_service.status(),
        },
        "metrics": {
            "enhancement_latency_ms": enhancement_ms,
            "detection_latency_ms": detection_ms,
            "total_latency_ms": total_ms,
            "fps_equivalent": round(1000 / total_ms, 2) if total_ms else 0,
        },
        "detections_count": len(detections),
        "detections": detections,
        "enhanced_image": _image_to_data_url(processed),
        "annotated_image": _image_to_data_url(annotated),
    }
