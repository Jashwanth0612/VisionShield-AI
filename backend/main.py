from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.benchmark import router as benchmark_router
from app.api.pipeline import nafnet_service, rtdetr_service, router as pipeline_router
from app.api.video import router as video_router
from app.core.config import settings
from app.services.storage import storage


@asynccontextmanager
async def lifespan(_: FastAPI):
    nafnet_service.load_model()
    rtdetr_service.load_model()
    yield


app = FastAPI(
    title=settings.app_name,
    description="All-weather image restoration with NAFNet and object detection with RT-DETR.",
    version=settings.version,
    lifespan=lifespan,
)

origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router)
app.include_router(benchmark_router)
app.include_router(video_router)


@app.get("/", tags=["System"])
def read_root():
    return {
        "status": "online",
        "system": "VisionShield AI Backend",
        "version": settings.version,
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
def health_check():
    ready = nafnet_service.loaded and rtdetr_service.loaded
    return {"status": "healthy" if ready else "degraded", "api_status": "connected"}


@app.get("/health/models", tags=["System"])
def model_health():
    nafnet = nafnet_service.status()
    rt_detr = rtdetr_service.status()
    ready = nafnet_service.loaded and rtdetr_service.loaded
    return {
        "status": "healthy" if ready else "degraded",
        "models": {"nafnet": nafnet, "rt_detr": rt_detr},
        "artifact_store": {"status": "ready", "provider": "replaceable-local"},
        "total_inferences": storage.inference_count(),
    }
