import io

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app.services.benchmark_service import benchmark_image
from app.services.storage import storage

router = APIRouter(prefix="/benchmark", tags=["Benchmark"])
VALID_WEATHER = {"auto", "fog_its", "fog_ots", "rain", "snow", "low_light"}


@router.post("/image")
async def benchmark_endpoint(
    file: UploadFile = File(...),
    runs: int = Query(3, ge=1, le=10),
    enhancement: bool = Query(True),
    confidence: float | None = Query(None, ge=0.05, le=0.99),
    weather: str = Query("auto"),
):
    weather = weather.lower().strip()
    if weather not in VALID_WEATHER:
        raise HTTPException(status_code=400, detail=f"Unsupported weather route. Choose one of: {', '.join(sorted(VALID_WEATHER))}.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB upload limit.")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image.") from exc
    if image.width * image.height > 25_000_000:
        raise HTTPException(status_code=413, detail="Image dimensions exceed the 25 MP limit.")
    try:
        result = benchmark_image(image, file.filename or "benchmark-image", runs, enhancement, confidence, weather)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc
    return {"status": "success", "result": result}


@router.get("/history")
def benchmark_history():
    return storage.list_benchmarks()


@router.get("/summary")
def benchmark_summary():
    return storage.benchmark_summary()
