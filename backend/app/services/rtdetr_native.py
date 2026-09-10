from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms
from torchvision.transforms import Compose, Resize, ToTensor


CLASS_NAMES = {
    0: "Vehicle",
    1: "Bus",
    2: "Motorcycle",
    3: "Truck",
}


def _runtime_root() -> Path:
    """Return the official RT-DETRv2 source checkout used by the backend."""
    configured = Path(__import__("os").getenv("RTDETR_RUNTIME_ROOT", "/opt/RT-DETR/rtdetrv2_pytorch"))
    if not configured.is_dir():
        raise RuntimeError(
            f"RT-DETRv2 runtime source is missing at {configured}. "
            "The backend image must clone the official rtdetrv2_pytorch runtime."
        )
    if str(configured) not in sys.path:
        sys.path.insert(0, str(configured))
    return configured


class AQMRTDETRTransformerv2:
    """Factory marker for the production AQM+SOS decoder patch."""


class NativeRTDETR:
    """Native PyTorch RT-DETRv2 AQM+SOS inference runtime.

    The backbone, HybridEncoder and transformer implementation are loaded from
    the official RT-DETRv2 PyTorch source. AQM and SOS are applied as inference
    logic only; they introduce no trainable parameters, so the project's
    RTDETRv2_AQM_SOS checkpoint remains structurally compatible.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.model: nn.Module | None = None
        self.loaded = False
        self.load_error: str | None = None

    def load(self) -> None:
        if self.loaded:
            return
        self.load_error = None
        try:
            _runtime_root()
            from src.nn.backbone.presnet import PResNet
            from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
            from src.zoo.rtdetr.rtdetr import RTDETR
            from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2

            class ProductionDecoder(RTDETRTransformerv2):
                """Exact AQM/SOS behavior used by the trained checkpoint."""

                def __init__(self, *args: Any, use_adaptive_query_masking: bool = False,
                             use_small_object_scoring: bool = True, **kwargs: Any) -> None:
                    super().__init__(*args, **kwargs)
                    self.use_adaptive_query_masking = use_adaptive_query_masking
                    self.use_small_object_scoring = use_small_object_scoring

                def get_adaptive_query_count(self, memory: torch.Tensor) -> int:
                    if not self.use_adaptive_query_masking:
                        return self.num_queries
                    activation = memory.abs().mean().item()
                    if activation < 0.10:
                        return self.num_queries
                    if activation < 0.25:
                        return max(200, self.num_queries // 2)
                    return max(100, self.num_queries // 3)

                def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor,
                                 outputs_coords_unact: torch.Tensor, topk: int):
                    if self.query_select_method == "default":
                        scores = outputs_logits.max(-1).values
                        if self.use_small_object_scoring:
                            boxes = torch.sigmoid(outputs_coords_unact)
                            wh = boxes[..., 2:]
                            area = wh[..., 0] * wh[..., 1]
                            small_object_weight = 1.0 / (area + 1e-2)
                            max_val = small_object_weight.max(dim=1, keepdim=True).values + 1e-6
                            small_object_weight = small_object_weight / max_val
                            final_score = scores + 0.2 * small_object_weight
                        else:
                            final_score = scores
                        _, topk_ind = torch.topk(final_score, topk, dim=-1)
                    elif self.query_select_method == "one2many":
                        _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
                        topk_ind = topk_ind // self.num_classes
                    elif self.query_select_method == "agnostic":
                        _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
                    else:
                        raise ValueError(f"Unsupported query selection method: {self.query_select_method}")

                    topk_coords = outputs_coords_unact.gather(
                        1, topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1])
                    )
                    topk_logits = outputs_logits.gather(
                        1, topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])
                    )
                    topk_memory = memory.gather(
                        1, topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
                    )
                    return topk_memory, topk_logits, topk_coords

                def _get_decoder_input(self, memory: torch.Tensor, spatial_shapes,
                                       denoising_logits=None, denoising_bbox_unact=None):
                    if self.training or self.eval_spatial_size is None:
                        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
                    else:
                        anchors = self.anchors
                        valid_mask = self.valid_mask

                    memory = valid_mask.to(memory.dtype) * memory
                    output_memory = self.enc_output(memory)
                    enc_outputs_logits = self.enc_score_head(output_memory)
                    enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

                    enc_topk_bboxes_list, enc_topk_logits_list = [], []
                    adaptive_queries = self.get_adaptive_query_count(memory)
                    enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = self._select_topk(
                        output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries
                    )

                    if self.use_adaptive_query_masking:
                        mask = torch.zeros(enc_topk_logits.shape[:2], device=enc_topk_logits.device)
                        mask[:, :adaptive_queries] = 1.0
                        enc_topk_memory = enc_topk_memory * mask.unsqueeze(-1)
                        enc_topk_logits = enc_topk_logits * mask.unsqueeze(-1)
                        enc_topk_bbox_unact = enc_topk_bbox_unact * mask.unsqueeze(-1)

                    if self.training:
                        enc_topk_bboxes_list.append(torch.sigmoid(enc_topk_bbox_unact))
                        enc_topk_logits_list.append(enc_topk_logits)

                    if self.learn_query_content:
                        content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
                    else:
                        content = enc_topk_memory.detach()
                    enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

                    if denoising_bbox_unact is not None:
                        enc_topk_bbox_unact = torch.concat(
                            [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
                        )
                        content = torch.concat([denoising_logits, content], dim=1)

                    return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

            backbone = PResNet(
                depth=50,
                variant="d",
                freeze_at=0,
                return_idx=[1, 2, 3],
                num_stages=4,
                freeze_norm=True,
                pretrained=False,
            )
            encoder = HybridEncoder(
                in_channels=[512, 1024, 2048],
                feat_strides=[8, 16, 32],
                hidden_dim=256,
                use_encoder_idx=[2],
                num_encoder_layers=1,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.0,
                enc_act="gelu",
                expansion=0.5,
                depth_mult=1.0,
                act="silu",
                eval_spatial_size=[640, 640],
            )
            decoder = ProductionDecoder(
                num_classes=8,
                feat_channels=[256, 256, 256],
                feat_strides=[8, 16, 32],
                hidden_dim=256,
                num_levels=3,
                num_layers=6,
                num_queries=300,
                num_denoising=100,
                label_noise_ratio=0.5,
                box_noise_scale=1.0,
                eval_spatial_size=[640, 640],
                eval_idx=2,
                num_points=[4, 4, 4],
                use_adaptive_query_masking=False,
                use_small_object_scoring=True,
            )
            model = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)

            # Memory-map the detector checkpoint so the serialized file is not
            # duplicated wholesale in RAM before state_dict loading.
            checkpoint = torch.load(
                self.checkpoint,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            state = {
                key: value for key, value in state.items()
                if "anchors" not in key and "valid_mask" not in key
            }
            missing, unexpected = model.load_state_dict(state, strict=False)
            critical_missing = [k for k in missing if "anchors" not in k and "valid_mask" not in k]
            if critical_missing or unexpected:
                raise RuntimeError(
                    "RT-DETR checkpoint architecture mismatch: "
                    f"missing={critical_missing[:8]}, unexpected={unexpected[:8]}"
                )
            del state
            del checkpoint
            gc.collect()

            model.to(self.device)
            model.eval()
            self.model = model
            self.loaded = True
        except Exception as exc:
            self.model = None
            self.loaded = False
            self.load_error = f"Unable to load native RT-DETRv2 AQM+SOS runtime: {exc}"

    @torch.inference_mode()
    def detect(self, image: Image.Image, confidence: float = 0.35) -> list[dict[str, Any]]:
        self.load()
        if not self.loaded or self.model is None:
            raise RuntimeError(self.load_error or "RT-DETRv2 model is unavailable.")

        original_width, original_height = image.size
        tensor = Compose([Resize((640, 640)), ToTensor()])(image.convert("RGB"))
        outputs = self.model(tensor.unsqueeze(0).to(self.device))
        logits = outputs["pred_logits"][0]
        boxes = outputs["pred_boxes"][0]
        scores = logits.softmax(-1)
        scores, labels = scores.max(-1)
        keep = scores > confidence
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if boxes.numel() == 0:
            return []

        xyxy = torch.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        keep = nms(xyxy, scores, 0.4)

        detections: list[dict[str, Any]] = []
        for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
            cx, cy, w, h = [float(v) for v in box.tolist()]
            detections.append({
                "bbox": [
                    round(max(0.0, (cx - w / 2) * original_width), 2),
                    round(max(0.0, (cy - h / 2) * original_height), 2),
                    round(min(float(original_width), (cx + w / 2) * original_width), 2),
                    round(min(float(original_height), (cy + h / 2) * original_height), 2),
                ],
                "confidence": round(float(score), 4),
                "class_id": int(label),
                "label": CLASS_NAMES.get(int(label), f"Class {int(label)}"),
            })
        return detections

    def status(self) -> dict[str, Any]:
        native_status = self.model.status() if self.model is not None else None
        return {
            "loaded": self.loaded,
            "checkpoint": self.checkpoint.name,
            "checkpoint_exists": self.checkpoint.is_file(),
            "device": str(self.device),
            "aqm": False,
            "sos": True,
            "error": self.load_error,
            "native": native_status,
        }
