from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

Weather = Literal["fog_its", "fog_ots", "rain", "snow", "low_light"]

LABELS: dict[str, str] = {
    "fog_its": "Fog · ITS",
    "fog_ots": "Fog · OTS",
    "rain": "Rain",
    "snow": "Snow",
    "low_light": "Low-Light",
}


def classify_weather(image: Image.Image) -> tuple[Weather, dict[str, float]]:
    """Lightweight rule-based routing before condition-specific NAFNet restoration.

    Ambiguous low-contrast scenes conservatively route to Fog-ITS. The API also
    accepts an explicit weather override, so operators do not have to rely on
    heuristic classification when the condition is known.
    """
    rgb = image.convert("RGB").resize((160, 160))
    arr = np.asarray(rgb, dtype=np.float32)
    gray = arr.mean(axis=2)
    contrast = float(gray.std())
    brightness = float(gray.mean())
    saturation = float(arr.max(axis=2).mean() - arr.min(axis=2).mean())

    if brightness < 55:
        weather: Weather = "low_light"
    elif brightness > 175 and saturation < 55 and contrast < 65:
        weather = "snow"
    elif contrast < 38 and saturation < 75:
        weather = "fog_its"
    elif saturation < 95 and contrast < 55:
        weather = "rain"
    else:
        weather = "fog_its"

    return weather, {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "saturation": round(saturation, 2),
    }


def weather_label(weather: str) -> str:
    return LABELS.get(weather, weather.replace("_", " ").title())
