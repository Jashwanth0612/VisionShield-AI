from __future__ import annotations

import io
import time
import uuid
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, UnidentifiedImageError

from app.services.image_utils import image_to_data_url
from app.services.nafnet_service import NAFNetService
from app.services.rtdetr_service import RTDETRService
from app.services.storage import storage

router = APIRouter(tags=["Pipeline"])
nafnet_service = NAFNetService()
rtdetr_service = RTDETRService()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def _annotate(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, min(8, image.width // 400))
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = f'{detection["label"]} {detection["confidence"]:.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=(0, 240, 255), width=line_width)
        try:
            bbox = draw.textbbox((x1, y1), label)
            draw.rectangle(bbox, fill=(8, 15, 20))
        except Exception:
            pass
        draw.text((x1 + 3, y1 + 2), label, fill=(248, 250, 252))
    return annotated


def _validate_image(file: UploadFile, contents: bytes) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    try:
        image = Image.open(io.BytesIO(contents))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image dimensions exceed the 25 MP safety limit.")
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image.") from exc


def model_health() -> dict[str, Any]:
    return {
        "nafnet": nafnet_service.status(),
        "rt_detr": rtdetr_service.status(),
    }


def _require_models(enable_enhancement: bool) -> None:
    if enable_enhancement and not nafnet_service.loaded:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": nafnet_service.load_error or "NAFNet is unavailable.", "model": "NAFNet"})
    if not rtdetr_service.loaded:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": rtdetr_service.load_error or "RT-DETR is unavailable.", "model": "RT-DETR"})


@router.post("/pipeline/process")
async def process_image(
    file: UploadFile = File(...),
    enable_enhancement: bool = Query(True),
    confidence: float | None = Query(None, ge=0.05, le=0.99),
):
    """Run real NAFNet -> RT-DETR inference and persist generated artifacts."""
    _require_models(enable_enhancement)
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB upload limit.")
    image = _validate_image(file, contents)

    run_id = f"inf_{uuid.uuid4().hex[:12]}"
    original_artifact = storage.save_artifact(contents, run_id, "original", file.filename, file.content_type)
    pipeline_start = time.perf_counter()

    enhancement_start = time.perf_counter()
    processed = nafnet_service.enhance_image(image) if enable_enhancement else image.copy()
    enhancement_ms = round((time.perf_counter() - enhancement_start) * 1000, 2)

    detection_start = time.perf_counter()
    detections = rtdetr_service.detect(processed, confidence_threshold=confidence)
    detection_ms = round((time.perf_counter() - detection_start) * 1000, 2)
    annotated = _annotate(processed, detections)
    total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
    fps = round(1000 / total_ms, 2) if total_ms else 0

    enhanced_buffer = io.BytesIO()
    processed.save(enhanced_buffer, format="PNG")
    annotated_buffer = io.BytesIO()
    annotated.save(annotated_buffer, format="PNG")
    enhanced_artifact = storage.save_artifact(enhanced_buffer.getvalue(), run_id, "enhanced", "enhanced.png", "image/png")
    annotated_artifact = storage.save_artifact(annotated_buffer.getvalue(), run_id, "annotated", "annotated.png", "image/png")

    models = model_health()
    record = {
        "run_id": run_id,
        "timestamp": storage.now(),
        "media_type": "image",
        "filename": file.filename or "image",
        "nafnet_enabled": enable_enhancement,
        "confidence_threshold": confidence if confidence is not None else rtdetr_service.conf_threshold,
        "detections": len(detections),
        "latency_ms": total_ms,
        "fps": fps,
        "model_status": "ready",
        "model_config": f"NAFNet={'on' if enable_enhancement else 'off'} · RT-DETR conf={confidence if confidence is not None else rtdetr_service.conf_threshold:.2f}",
        "artifacts": {"original": original_artifact, "enhanced": enhanced_artifact if enable_enhancement else None, "annotated": annotated_artifact},
        "details": {"image_size": {"width": image.width, "height": image.height}, "enhancement_latency_ms": enhancement_ms, "detection_latency_ms": detection_ms, "detections_detail": detections, "models": models},
    }
    storage.record_inference(record)

    return {
        "status": "success",
        "run_id": run_id,
        "timestamp": record["timestamp"],
        "filename": record["filename"],
        "image_size": {"width": image.width, "height": image.height},
        "enhancement_enabled": enable_enhancement,
        "models": models,
        "metrics": {"enhancement_latency_ms": enhancement_ms, "detection_latency_ms": detection_ms, "total_latency_ms": total_ms, "fps_equivalent": fps},
        "detections_count": len(detections),
        "detections": detections,
        "artifacts": record["artifacts"],
        "original_image": image_to_data_url(image),
        "enhanced_image": image_to_data_url(processed) if enable_enhancement else None,
        "annotated_image": image_to_data_url(annotated),
    }


@router.get("/health")
def health():
    models = model_health()
    ready = rtdetr_service.loaded and nafnet_service.loaded
    return {
        "status": "operational" if ready else "degraded",
        "api_status": "connected",
        "models": models,
        "artifact_store": {"status": "ready", "provider": "replaceable-local"},
        "total_inferences": storage.inference_count(),
    }


@router.get("/history")
def history(search: str = "", media_type: str = "all", limit: int = 50):
    return storage.list_inferences(search=search, media_type=media_type, limit=limit)


@router.get("/history/{run_id}")
def history_item(run_id: str):
    item = storage.get_inference(run_id)
    if not item:
        raise HTTPException(status_code=404, detail="Inference history item not found")
    return item


@router.get("/artifacts/{artifact_id}")
def artifact(artifact_id: str):
    path = storage.artifact_path(artifact_id)
    if not path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)
