from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
import torchvision.transforms as transforms

from app.core.config import settings


class NAFNetService:
    """NAFNet adapter.

    The exact NAFNet architecture/checkpoint is project-specific, so this service
    intentionally loads a TorchScript or serialized module when supplied rather
    than pretending an arbitrary checkpoint is compatible with a guessed model.
    """

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = Path(model_path or settings.nafnet_checkpoint)
        self.model: Any = None
        self.loaded = False
        self.load_error: str | None = None
        self.transform = transforms.ToTensor()

    def load_model(self) -> None:
        if not self.model_path.exists():
            self.loaded = False
            self.load_error = "NAFNet checkpoint not found; enhancement will use pass-through mode."
            return

        try:
            # Supports a TorchScript module or a fully serialized nn.Module.
            try:
                self.model = torch.jit.load(str(self.model_path), map_location=self.device)
            except RuntimeError:
                self.model = torch.load(str(self.model_path), map_location=self.device, weights_only=False)

            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.loaded = False
            self.load_error = str(exc)

    @torch.inference_mode()
    def enhance_image(self, image: Image.Image) -> Image.Image:
        if self.model is None:
            return image.copy()

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output.squeeze(0).detach().cpu().clamp(0, 1)
        return transforms.ToPILImage()(output)

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "checkpoint": str(self.model_path),
            "checkpoint_exists": self.model_path.exists(),
            "device": str(self.device),
            "error": self.load_error,
        }
