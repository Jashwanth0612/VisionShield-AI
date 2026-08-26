# VisionShield AI

> **All-weather computer vision pipeline for image enhancement and robust object detection.**

VisionShield AI combines **NAFNet-based image restoration** with **RT-DETR object detection** to improve perception in degraded visual conditions such as fog, rain, low light, and other challenging environments.

## Pipeline

```text
Input Image
    |
    v
NAFNet Enhancement
    |
    v
RT-DETR Detection
    |
    +--> Bounding Boxes + Confidence
    +--> Latency / FPS Metrics
    +--> Annotated Output
```

## Current backend

- FastAPI inference API
- Configurable NAFNet checkpoint adapter
- RT-DETR inference through Ultralytics
- Automatic fallback to pretrained RT-DETR when a local checkpoint is absent
- Annotated detection output returned as a data URL
- Enhancement, detection, total-latency and FPS-equivalent metrics
- Model health endpoint
- Docker support

## API

Start locally:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `http://localhost:8000/docs`.

### Main endpoints

- `GET /` — service information
- `GET /health` — API health
- `GET /health/models` — model loading status
- `POST /pipeline/process` — enhancement + RT-DETR detection

Example:

```bash
curl -X POST "http://localhost:8000/pipeline/process" \
  -F "file=@sample.jpg" \
  -F "enable_enhancement=true" \
  -F "confidence=0.35"
```

## Model weights

Place project-specific weights at:

```text
models/
├── nafnet/
│   └── nafnet_weights.pth
└── rt_detr/
    └── rtdetr_weights.pt
```

The repository intentionally does **not** commit large model binaries. NAFNet loading expects a compatible serialized/TorchScript module; the exact NAFNet architecture should match the checkpoint you trained or obtained during the research project.

## Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## Roadmap

- [x] FastAPI backend foundation
- [x] RT-DETR inference adapter
- [x] NAFNet checkpoint adapter
- [x] Annotated inference response
- [x] Dockerized backend
- [ ] Production web dashboard
- [ ] Image comparison slider
- [ ] Video inference
- [ ] Experiment benchmarking
- [ ] Authentication and project history
- [ ] Cloud deployment

## Tech stack

**Python · FastAPI · PyTorch · TorchVision · Ultralytics RT-DETR · OpenCV · Pillow · Docker**

---

Built as an AI/ML computer-vision portfolio project focused on practical deployment and edge-oriented perception.
