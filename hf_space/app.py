from __future__ import annotations

# ZeroGPU must be imported before torch-backed model modules.
import spaces
import gradio as gr
import io
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError
from gradio import FileData

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.image_utils import image_to_data_url
from app.services.nafnet_service import NAFNetService, WEATHER_LABELS
from app.services.rtdetr_service import RTDETRService
from app.services.storage import storage
from app.services.weather_classifier import classify_weather, weather_label

VALID_WEATHER = {"auto", *WEATHER_LABELS.keys()}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

# ZeroGPU provides CUDA emulation during startup. Keeping model placement at module
# scope lets ZeroGPU pack the weights once and stream them into the real GPU worker.
# We keep one NAFNet service per condition so switching weather does not reload a
# checkpoint during the user's GPU allocation.
NAFNET_SERVICES: dict[str, NAFNetService] = {}
NAFNET_ERRORS: dict[str, str] = {}
for _weather in WEATHER_LABELS:
    _service = NAFNetService(device="cuda")
    try:
        _service.load_model(_weather)
    except Exception as exc:  # defensive: the service itself records normal load errors
        NAFNET_ERRORS[_weather] = str(exc)
    if _service.loaded:
        NAFNET_SERVICES[_weather] = _service
    elif _service.load_error:
        NAFNET_ERRORS[_weather] = _service.load_error

RTDETR_SERVICE = RTDETRService()
try:
    RTDETR_SERVICE.device = "cuda"
    RTDETR_SERVICE.load_model()
except Exception as exc:
    RTDETR_SERVICE.load_error = str(exc)


def _file_path(file: FileData) -> Path:
    path = getattr(file, "path", None)
    if not path and isinstance(file, dict):
        path = file.get("path")
    if not path:
        raise ValueError("No uploaded file was received.")
    return Path(path)


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


def _load_image(file: FileData) -> tuple[bytes, Image.Image, str]:
    path = _file_path(file)
    contents = path.read_bytes()
    if not contents:
        raise ValueError("Uploaded image is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 20 MB upload limit.")
    try:
        image = Image.open(io.BytesIO(contents))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions exceed the 25 MP safety limit.")
        return contents, image.convert("RGB"), path.name
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid or unsupported image.") from exc


def _select_weather(image: Image.Image, weather: str) -> tuple[str, dict[str, float] | None]:
    weather = weather.lower().strip()
    if weather not in VALID_WEATHER:
        raise ValueError(f"Unsupported weather route. Choose one of: {', '.join(sorted(VALID_WEATHER))}.")
    if weather == "auto":
        return classify_weather(image)
    return weather, None


def _model_status() -> dict[str, Any]:
    nafnet = {
        "loaded": bool(NAFNET_SERVICES),
        "configured": bool(NAFNET_SERVICES),
        "device": "cuda",
        "conditions": {
            weather: {
                "label": WEATHER_LABELS[weather],
                "configured": weather in NAFNET_SERVICES,
                "loaded": weather in NAFNET_SERVICES,
                "error": NAFNET_ERRORS.get(weather),
            }
            for weather in WEATHER_LABELS
        },
    }
    return {"nafnet": nafnet, "rt_detr": RTDETR_SERVICE.status()}


@spaces.GPU(duration=60)
def process_image(
    file: FileData,
    enable_enhancement: bool = True,
    confidence: float = 0.35,
    weather: str = "auto",
) -> dict[str, Any]:
    """Run weather routing, condition-specific NAFNet restoration, and native RT-DETRv2 detection."""
    contents, image, filename = _load_image(file)
    selected_weather, weather_features = _select_weather(image, weather)

    if not RTDETR_SERVICE.loaded:
        raise RuntimeError(RTDETR_SERVICE.load_error or "Native RT-DETRv2 is unavailable.")

    nafnet = NAFNET_SERVICES.get(selected_weather)
    if enable_enhancement and nafnet is None:
        raise RuntimeError(NAFNET_ERRORS.get(selected_weather) or f"NAFNet is unavailable for {weather_label(selected_weather)}.")

    run_id = f"inf_{uuid.uuid4().hex[:12]}"
    original_artifact = storage.save_artifact(contents, run_id, "original", filename, "image/*")
    started = time.perf_counter()

    enhancement_start = time.perf_counter()
    processed = nafnet.enhance_image(image, selected_weather) if enable_enhancement else image.copy()
    enhancement_ms = round((time.perf_counter() - enhancement_start) * 1000, 2)

    detection_start = time.perf_counter()
    detections = RTDETR_SERVICE.detect(processed, confidence_threshold=float(confidence))
    detection_ms = round((time.perf_counter() - detection_start) * 1000, 2)
    annotated = _annotate(processed, detections)
    total_ms = round((time.perf_counter() - started) * 1000, 2)
    fps = round(1000 / total_ms, 2) if total_ms else 0

    enhanced_buffer = io.BytesIO()
    processed.save(enhanced_buffer, format="PNG")
    annotated_buffer = io.BytesIO()
    annotated.save(annotated_buffer, format="PNG")
    enhanced_artifact = storage.save_artifact(enhanced_buffer.getvalue(), run_id, "enhanced", "enhanced.png", "image/png")
    annotated_artifact = storage.save_artifact(annotated_buffer.getvalue(), run_id, "annotated", "annotated.png", "image/png")

    models = _model_status()
    timestamp = storage.now()
    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "media_type": "image",
        "filename": filename,
        "nafnet_enabled": enable_enhancement,
        "weather_route": selected_weather,
        "weather_label": weather_label(selected_weather),
        "weather_source": "rule-based-auto" if weather == "auto" else "operator-selected",
        "weather_features": weather_features,
        "confidence_threshold": float(confidence),
        "detections": len(detections),
        "latency_ms": total_ms,
        "fps": fps,
        "model_status": "ready",
        "model_config": f"NAFNet={weather_label(selected_weather)} {'on' if enable_enhancement else 'off'} · RT-DETR conf={float(confidence):.2f}",
        "artifacts": {"original": original_artifact, "enhanced": enhanced_artifact if enable_enhancement else None, "annotated": annotated_artifact},
        "details": {
            "image_size": {"width": image.width, "height": image.height},
            "enhancement_latency_ms": enhancement_ms,
            "detection_latency_ms": detection_ms,
            "detections_detail": detections,
            "models": models,
        },
    }
    storage.record_inference(record)

    return {
        "status": "success",
        "run_id": run_id,
        "timestamp": timestamp,
        "filename": filename,
        "image_size": {"width": image.width, "height": image.height},
        "enhancement_enabled": enable_enhancement,
        "weather": {
            "requested": weather,
            "selected": selected_weather,
            "label": weather_label(selected_weather),
            "source": record["weather_source"],
            "features": weather_features,
        },
        "models": models,
        "metrics": {
            "enhancement_latency_ms": enhancement_ms,
            "detection_latency_ms": detection_ms,
            "total_latency_ms": total_ms,
            "fps_equivalent": fps,
        },
        "detections_count": len(detections),
        "detections": detections,
        "artifacts": record["artifacts"],
        "original_image": image_to_data_url(image),
        "enhanced_image": image_to_data_url(processed) if enable_enhancement else None,
        "annotated_image": image_to_data_url(annotated),
    }


def health() -> dict[str, Any]:
    """Return runtime and model availability without allocating GPU time."""
    models = _model_status()
    ready = bool(RTDETR_SERVICE.loaded and NAFNET_SERVICES)
    return {
        "status": "healthy" if ready else "degraded",
        "api_status": "connected",
        "models_loaded": ready,
        "runtime_configured": bool(NAFNET_SERVICES or RTDETR_SERVICE.model_path),
        "model_loading": "startup-preloaded",
        "models": models,
        "total_inferences": storage.inference_count(),
    }


def history(search: str = "", media_type: str = "all", limit: int = 50) -> list[dict[str, Any]]:
    """Return stored inference history."""
    return storage.list_inferences(search=search, media_type=media_type, limit=limit)


def history_item(run_id: str) -> dict[str, Any]:
    """Return one stored inference history item."""
    item = storage.get_inference(run_id)
    if not item:
        raise ValueError("Inference history item not found")
    return item


with gr.Blocks(title="VisionShield AI Runtime") as demo:
    gr.Markdown(
        "# VisionShield AI Runtime\n"
        "All-weather visual perception powered by condition-specific NAFNet restoration and native RT-DETRv2 detection.\n\n"
        "The production React client uses the named Gradio API endpoints below."
    )
    gr.Markdown("**API:** `/gradio_api` · **Runtime:** ZeroGPU · **Models:** NAFNet × 5 + RT-DETRv2")

    gr.api(process_image, api_name="process_image", api_description="Run the full VisionShield image perception pipeline.")
    gr.api(health, api_name="health", api_description="Return runtime and model availability.")
    gr.api(history, api_name="history", api_description="Return stored inference history.")
    gr.api(history_item, api_name="history_item", api_description="Return one stored inference history item.")

if __name__ == "__main__":
    demo.launch()
