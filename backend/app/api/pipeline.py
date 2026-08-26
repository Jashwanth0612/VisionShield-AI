import io
import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image

from app.services.nafnet_service import NAFNetService
from app.services.rtdetr_service import RTDETRService

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# Initialize services
nafnet_service = NAFNetService()
rtdetr_service = RTDETRService()

# Load model weights on startup
nafnet_service.load_model()
rtdetr_service.load_model()


@router.post("/process")
async def process_image(file: UploadFile = File(...), enable_enhancement: bool = True):
    """
    Process image through NAFNet enhancement followed by RT-DETR object detection.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    start_time = time.time()

    # Step 1: All-Weather Image Enhancement (NAFNet)
    enhancement_start = time.time()
    if enable_enhancement:
        processed_image = nafnet_service.enhance_image(image)
    else:
        processed_image = image
    enhancement_time = round((time.time() - enhancement_start) * 1000, 2)

    # Step 2: Object Detection (RT-DETR)
    detection_start = time.time()
    detections = rtdetr_service.detect(processed_image)
    detection_time = round((time.time() - detection_start) * 1000, 2)

    total_time = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "success",
        "filename": file.filename,
        "image_size": image.size,
        "enhancement_enabled": enable_enhancement,
        "pipeline_metrics": {
            "enhancement_latency_ms": enhancement_time,
            "detection_latency_ms": detection_time,
            "total_latency_ms": total_time
        },
        "detections_count": len(detections),
        "detections": detections
    }
