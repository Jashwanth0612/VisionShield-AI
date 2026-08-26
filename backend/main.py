from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.pipeline import nafnet_service, rtdetr_service, router as pipeline_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    nafnet_service.load_model()
    rtdetr_service.load_model()
    yield


app = FastAPI(
    title=settings.app_name,
    description="All-weather image enhancement with NAFNet and object detection with RT-DETR.",
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
    return {"status": "healthy"}


@app.get("/health/models", tags=["System"])
def model_health():
    return {
        "status": "healthy",
        "models": {
            "nafnet": nafnet_service.status(),
            "rt_detr": rtdetr_service.status(),
        },
    }
