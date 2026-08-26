# VisionShield AI

> **Premium all-weather computer-vision console combining condition-specific NAFNet restoration and RT-DETRv2 detection.**

VisionShield AI is split into a presentation layer, a clean API service boundary, and replaceable backend model/storage services. The UI never fabricates inference, benchmark, history, or artifact data.

## Research pipeline

The project presentation defines the end-to-end flow as:

```text
Input image / camera
        |
        v
Rule-based weather routing
        |
        +--> Fog · ITS NAFNet
        +--> Fog · OTS NAFNet
        +--> Rain NAFNet
        +--> Snow NAFNet
        +--> Low-Light NAFNet
        |
        v
Enhanced image
        |
        v
RT-DETRv2 + AQM + SOS
        |
        v
Bounding boxes + labels + confidence
```

The presentation describes five separate NAFNet models for Fog-ITS, Fog-OTS, Rain, Snow and Low-Light, with weather selection at inference time. It also documents auto-resizing images above 1024 px before NAFNet and restoring the original size afterward.

## Architecture reference

### NAFNet

The documented training configuration is **width=64**, `enc_blk=[1,1,1,28]`, `middle=1`, `dec_blk=[1,1,1,1]`, AdamW, cosine-annealed learning rate from `1e-3` to `1e-6`, 15,000 iterations, and 256×256 patches.

Reported restoration results in the project presentation:

| Dataset / condition | Train | Validation | PSNR | SSIM |
|---|---:|---:|---:|---:|
| RESIDE ITS (Fog) | 13,990 | 500 | 33.71 dB | 0.983 |
| RESIDE OTS (Fog) | 72,135 | 500 | 31.62 dB | 0.977 |
| Rain13K | 13,711 | 1,200 | 34.62 dB | 0.969 |
| Snow100K | 15,801 | 1,000 | 26.94 dB | 0.905 |
| LOL (Low-Light) | ~485 | 15 | 29.87 dB | 0.934 |

### RT-DETRv2

The documented detector uses a PResNet-50 backbone, HybridEncoder with AIFI + CCFM, a 6-layer transformer decoder with 300 queries and 3 detection levels, plus the project's Adaptive Query Masking (AQM) and Small Object Scoring (SOS) modifications.

The presentation reports baseline RT-DETRv2 evaluation on UA-DETRAC using COCO evaluation:

- **mAP@50:95:** 0.866
- **mAP@50:** 0.978
- **Recall:** 0.895
- **AP_small:** 0.0129 → 0.0196 (+52%) with AQM + SOS

The presentation also shows fog examples where enhancement increases visible detections from **4 to 8+** and confidence from roughly **0.72 to 0.90**. These are qualitative/example-scene observations, not a replacement for a dataset-level accuracy evaluation.

## Architecture

```text
React / Vite console
        |
        v
frontend/src/api.js
        |
        v
FastAPI
  |          |             |
  v          v             v
Weather   RT-DETRv2     StorageService
Router       |              |
  |          |              +--> SQLite metadata
  v          |              +--> Local artifacts
NAFNet       |
  |
  +----------+

StorageService is intentionally replaceable with MongoDB/S3/object storage later.
```

## Features

- Image inference with five-condition NAFNet routing and RT-DETR confidence threshold
- Auto weather routing plus explicit Fog-ITS, Fog-OTS, Rain, Snow and Low-Light selection
- Interactive bounding-box overlay plus original / enhanced / annotated views
- Real generated image artifacts
- Sampled video inference with generated enhanced and annotated result videos
- Persistent inference history with search, type filters, configuration, metrics and artifact links
- Explicit benchmark actions stored separately from inference history
- Measured latency/FPS/detection trends without inventing evaluation metrics
- Honest model unavailable / loading / error / empty states
- Responsive dark technical console suitable for a portfolio demonstration

## Model runtime

Model weights are **never committed** and are **never downloaded automatically**.

Set the five NAFNet checkpoints and the RT-DETR checkpoint before enabling inference:

```bash
NAFNET_ITS_WEIGHTS_PATH=/models/nafnet_its_best.pth
NAFNET_OTS_WEIGHTS_PATH=/models/nafnet_ots_best.pth
NAFNET_RAIN_WEIGHTS_PATH=/models/nafnet_rain_best.pth
NAFNET_SNOW_WEIGHTS_PATH=/models/nafnet_snow_best.pth
NAFNET_LOWLIGHT_WEIGHTS_PATH=/models/nafnet_lowlight_best.pth
RTDETR_WEIGHTS_PATH=/models/RTDETRv2_AQM_SOS_best.pth
```

The legacy `NAFNET_WEIGHTS_PATH` variable remains as a compatibility fallback for a single checkpoint and routes that checkpoint to Fog-ITS.

The NAFNet architecture parameters must match the supplied checkpoints:

```bash
NAFNET_WIDTH=64
NAFNET_MIDDLE_BLOCKS=1
NAFNET_ENCODER_BLOCKS=1,1,1,28
NAFNET_DECODER_BLOCKS=1,1,1,1
```

If paths are absent or incompatible, `/health` reports a degraded model runtime and inference returns `503 model_unavailable`. No synthetic detections or fabricated metrics are returned.

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
- `POST /pipeline/process` — weather routing → NAFNet → RT-DETRv2 inference
- `POST /video/analyze` — sampled real video inference
- `GET /pipeline/history` — persisted inference history
- `GET /pipeline/history/{run_id}` — persisted run detail
- `GET /artifacts/{artifact_id}` — generated media artifact
- `POST /benchmark/image` — explicit measured benchmark action with weather routing
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

## Measurement rule

The UI's live benchmark charts report only measured pipeline performance returned by the backend. Research metrics such as **PSNR, SSIM, mAP and Recall** shown above are documented project evaluation results; they are not regenerated from a live inference request unless a ground-truth evaluation endpoint is added.

## Tech stack

**React · Vite · FastAPI · PyTorch · TorchVision · OpenCV · Pillow · SQLite · Docker**
