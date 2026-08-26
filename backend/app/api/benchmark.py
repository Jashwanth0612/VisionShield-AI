from __future__ import annotations

import io

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app.services.benchmark_service import benchmark_image

router = APIRouter(prefix="/benchmark", tags=["Benchmark"])


@router.post("/image")
async def benchmark_endpoint(
    file: UploadFile = File(...),
    runs: int = Query(3, ge=1, le=10),
    enhancement: bool = Query(True),
):
    """Benchmark inference latency for an uploaded image."""
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
    return {"status": "success", "filename": file.filename, "result": benchmark_image(image, runs, enhancement)}
