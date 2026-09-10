---
title: VisionShield AI Runtime
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.1.0
python_version: 3.10.13
app_file: app.py
startup_duration_timeout: 30m
---

# VisionShield AI Runtime

ZeroGPU runtime adapter for the VisionShield AI all-weather perception pipeline.

The production React frontend remains separate. This Space exposes the ML runtime through Gradio API endpoints so the frontend can call the real NAFNet + native RT-DETRv2 pipeline without requiring a paid persistent GPU server.
