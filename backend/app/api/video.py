from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.services.video_service import MAX_VIDEO_BYTES, analyze_video

router = APIRouter(prefix="/video", tags=["Video"])

ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo", "video/mpeg"}
VALID_WEATHER = {"auto", "fog_its", "fog_ots", "rain", "snow", "low_light"}


@router.post("/analyze")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    sample_fps: float = Query(2.0, ge=0.5, le=10.0),
    enable_enhancement: bool = Query(True),
    confidence: float | None = Query(None, ge=0.05, le=0.99),
    weather: str = Query("auto"),
):
    """Analyze sampled video frames through weather-routed NAFNet + RT-DETR inference."""
    weather = weather.lower().strip()
    if weather not in VALID_WEATHER:
        raise HTTPException(status_code=400, detail=f"Unsupported weather route. Choose one of: {', '.join(sorted(VALID_WEATHER))}.")
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported video format. Use MP4, MOV, WebM, AVI, or MPEG.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")
    if len(contents) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video exceeds the configured upload limit.")
    try:
        result = analyze_video(contents, file.filename or "video.mp4", sample_fps, enable_enhancement, confidence, weather)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "success", "filename": file.filename, "result": result}
