# VisionShield AI

> **Premium all-weather computer-vision console combining NAFNet restoration and RT-DETR detection.**

VisionShield AI is split into a presentation layer, a clean API service boundary, and replaceable backend model/storage services. The UI never fabricates inference, benchmark, history, or artifact data.

## Architecture

```text
React / Vite console
        |
        v
frontend/src/api.js
        |
        v
FastAPI
  |       |        |
  v       v        v
NAFNet  RT-DETR  StorageService
  |       |        |
  +-------+        +--> SQLite metadata
                   +--> Local artifacts

StorageService is intentionally replaceable with MongoDB/S3/object storage later.
```

## Features

- Image inference with NAFNet toggle and RT-DETR confidence threshold
- Interactive bounding-box overlay plus original / enhanced / annotated views
- Real generated image artifacts
- Sampled video inference with generated enhanced and annotated result videos
- Persistent inference history with search, type filters, configuration, metrics and artifact links
- Explicit benchmark actions stored separately from inference history
- Measured latency/FPS/detection trends without accuracy claims
- Honest model unavailable / loading / error / empty states
- Responsive dark technical console suitable for a portfolio demonstration

## Model runtime

Model weights are **never committed** and are **never downloaded automatically**.

Set these environment variables before enabling inference:

```bash
NAFNET_WEIGHTS_PATH=/absolute/path/to/nafnet_weights.pth
RTDETR_WEIGHTS_PATH=/absolute/path/to/rtdetr_weights.pt
```

The RT-DETR service loads the supplied local checkpoint through Ultralytics. It deliberately does not fall back to a downloadable pretrained checkpoint.

The NAFNet service implements the official NAFNet architecture family and loads a compatible PyTorch checkpoint/state dictionary. The architecture parameters are configurable through `NAFNET_WIDTH`, `NAFNET_MIDDLE_BLOCKS`, `NAFNET_ENCODER_BLOCKS`, and `NAFNET_DECODER_BLOCKS`; they must match the supplied checkpoint. NAFNet publishes multiple task-specific checkpoints/configurations, so the checkpoint and architecture must be selected together.

Example:

```bash
export NAFNET_WEIGHTS_PATH=/models/nafnet/NAFNet-SIDD-width64.pth
export RTDETR_WEIGHTS_PATH=/models/rt_detr/rtdetr_custom.pt
export NAFNET_WIDTH=64
export NAFNET_MIDDLE_BLOCKS=1
export NAFNET_ENCODER_BLOCKS=1,1,1,28
export NAFNET_DECODER_BLOCKS=1,1,1,1
```

If the paths are absent or incompatible, `/health` reports `degraded` and inference returns `503 model_unavailable`. No synthetic detections or metrics are returned.

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

API documentation is available at `http://localhost:8000/docs`.

### Main endpoints

- `GET /` — service information
- `GET /health` — overall API/model readiness
- `GET /health/models` — detailed model and storage status
- `GET /pipeline/health` — dashboard health contract
- `POST /pipeline/process` — real image NAFNet + RT-DETR inference
- `POST /video/analyze` — sampled real video inference
- `GET /pipeline/history` — persisted inference history
- `GET /pipeline/history/{run_id}` — persisted run detail
- `GET /artifacts/{artifact_id}` — generated media artifact
- `POST /benchmark/image` — explicit measured benchmark action
- `GET /benchmark/summary` — benchmark trends and summary

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` if the backend is not at `http://localhost:8000`.

## Docker

```bash
docker compose up --build
```

Mount model files into `./models` and keep them out of Git. The compose setup persists application metadata and artifacts in a named volume.

## Storage boundary

The current `StorageService` uses SQLite for metadata and local disk for artifacts because it is simple and production-testable. It is deliberately isolated behind a small interface so a future deployment can replace it with MongoDB metadata plus S3-compatible object storage without redesigning the frontend.

## Important measurement rule

Benchmark charts report only measured pipeline performance returned by the backend. They are not accuracy, mAP, PSNR, SSIM, or other evaluation claims unless a future ground-truth evaluation endpoint explicitly supplies those metrics.

## Tech stack

**React · Vite · FastAPI · PyTorch · TorchVision · Ultralytics RT-DETR · OpenCV · Pillow · SQLite · Docker**
