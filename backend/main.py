from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="VisionShield AI API",
    description="All-weather image enhancement (NAFNet) and object detection (RT-DETR) pipeline.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "system": "VisionShield AI Backend"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
