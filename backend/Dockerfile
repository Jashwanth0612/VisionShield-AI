FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RTDETR_RUNTIME_ROOT=/opt/RT-DETR/rtdetrv2_pytorch \
    PYTHONPATH=/opt/RT-DETR/rtdetrv2_pytorch

WORKDIR /app

# OpenCV runtime libraries plus git for the official RT-DETRv2 source runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Pin the detector runtime to the official RT-DETR repository rather than
# duplicating the upstream source tree inside VisionShield.
RUN git clone --depth 1 https://github.com/lyuwenyu/RT-DETR.git /opt/RT-DETR

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
