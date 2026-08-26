from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from app.core.config import settings


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / torch.sqrt(var + eps)
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, -1, 1, 1) * y + bias.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, -1, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = (g - y * mean_gy - mean_g) / torch.sqrt(var + ctx.eps)
        return gx, (grad_output * y).sum(dim=(0, 2, 3)), grad_output.sum(dim=(0, 2, 3)), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True))
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.norm2 = LayerNorm2d(c)
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=None, dec_blk_nums=None):
        super().__init__()
        enc_blk_nums = enc_blk_nums or [1, 1, 1, 28]
        dec_blk_nums = dec_blk_nums or [1, 1, 1, 1]
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.encoders[-1].add_module("down", nn.Identity())
            chan *= 2
        # Build the encoder/decoder explicitly so checkpoint keys remain stable.
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, inp):
        _, _, h, w = inp.shape
        padded = self.check_image_size(inp)
        x = self.intro(padded)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        return (self.ending(x) + padded)[:, :, :h, :w]


WEATHER_LABELS = {
    "fog_its": "Fog · ITS",
    "fog_ots": "Fog · OTS",
    "rain": "Rain",
    "snow": "Snow",
    "low_light": "Low-Light",
}


class NAFNetService:
    """Condition-specific NAFNet inference with lazy checkpoint loading."""

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_paths: dict[str, Path] = settings.nafnet_paths
        self.models: dict[str, nn.Module] = {}
        self.load_errors: dict[str, str] = {}
        self.transform = transforms.ToTensor()

    def _build_model(self) -> NAFNet:
        return NAFNet(
            width=settings.nafnet_width,
            middle_blk_num=settings.nafnet_middle_blocks,
            enc_blk_nums=settings.encoder_blocks,
            dec_blk_nums=settings.decoder_blocks,
        )

    def _load_checkpoint(self, weather: str) -> None:
        if weather in self.models or weather in self.load_errors:
            return
        path = self.model_paths.get(weather)
        if path is None:
            self.load_errors[weather] = f"No NAFNet checkpoint configured for {WEATHER_LABELS.get(weather, weather)}."
            return
        if not path.is_file():
            self.load_errors[weather] = f"NAFNet checkpoint was not found: {path}"
            return
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(checkpoint, nn.Module):
                model = checkpoint
            else:
                state = checkpoint.get("params") or checkpoint.get("state_dict") or checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
                if not isinstance(state, dict):
                    raise ValueError("Checkpoint does not contain a PyTorch state_dict.")
                model = self._build_model()
                state = {key.replace("module.", "", 1): value for key, value in state.items()}
                model.load_state_dict(state, strict=True)
            self.models[weather] = model.to(self.device).eval()
        except Exception as exc:
            self.load_errors[weather] = f"Unable to load checkpoint: {exc}"

    def load_model(self, weather: str = "fog_its") -> None:
        self._load_checkpoint(weather)

    @property
    def loaded(self) -> bool:
        return bool(self.models)

    @property
    def load_error(self) -> str | None:
        return next(iter(self.load_errors.values()), None)

    @torch.inference_mode()
    def enhance_image(self, image: Image.Image, weather: str = "fog_its") -> Image.Image:
        self._load_checkpoint(weather)
        model = self.models.get(weather)
        if model is None:
            raise RuntimeError(self.load_errors.get(weather) or "NAFNet model is unavailable.")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output.squeeze(0).detach().cpu().clamp(0, 1)
        return transforms.ToPILImage()(output)

    def status(self) -> dict[str, Any]:
        conditions = {}
        for weather, path in self.model_paths.items():
            conditions[weather] = {
                "label": WEATHER_LABELS.get(weather, weather),
                "configured": bool(path),
                "checkpoint": path.name if path else None,
                "checkpoint_exists": bool(path and path.is_file()),
                "loaded": weather in self.models,
                "error": self.load_errors.get(weather),
            }
        return {
            "loaded": self.loaded,
            "configured": bool(self.model_paths),
            "device": str(self.device),
            "conditions": conditions,
            "error": self.load_error,
        }
