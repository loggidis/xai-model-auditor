import time

import numpy as np
import torch
from torchvision.ops import nms

from app.xai.gradcam_pp import _TensorOutput
from app.xai.inference import Detection
from app.xai.loader import LoadedModel
from app.xai.preprocess import letterbox

TARGET_PARAMS = {
    "iou": {"type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.7,
            "description": "NMS IoU for selecting the target detections"},
    "max_det": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20,
                "description": "Maximum number of detections in the target"},
    "detection_index": {"type": ["integer", "null"], "minimum": 0, "default": None,
                        "description": "null = all detections, k = only the k-th (by descending conf)"},
}


class XAITimeoutError(Exception):
    """The method exceeded its time limit."""


def _xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.cat([b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2], dim=-1)


def _iou(boxes: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """boxes (K,4), pred (B,A,4) -> IoU (B,K,A)."""
    b = boxes[None, :, None, :]
    p = pred[:, None, :, :]
    lt = torch.maximum(b[..., :2], p[..., :2])
    rb = torch.minimum(b[..., 2:], p[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    return inter / (area_b + area_p - inter + 1e-9)


class DetectionTargetAdapter:
    """Converts a detector into a function  image -> scalar,  for IntGrad / SHAP / LIME.

    1. On the clean image the targets are selected (conf threshold + NMS), as in Grad-CAM++.
    2. score(x) = Σ_targets  max { class score of the target over anchors with IoU >= match_iou with its box }
    """

    def __init__(self, model: LoadedModel, image_rgb: np.ndarray, conf_threshold: float,
                 iou: float = 0.7, max_det: int = 20, detection_index: int | None = None,
                 match_iou: float = 0.5):
        self.model = model
        self.device = model.device
        self.canvas, self.lb = letterbox(image_rgb, model.info.input_size)   # uint8 (S, S, 3)
        self.match_iou = match_iou
        self._net = _TensorOutput(model.net, model.info.num_classes)
        self._deadline: float | None = None
        self._timeout = 0.0

        x = torch.from_numpy(self.canvas).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._net(x)[0]                                    # (4+nc, A)
        conf, cls = out[4:].max(dim=0)
        idx = (conf >= conf_threshold).nonzero(as_tuple=True)[0]
        self.boxes = torch.empty((0, 4), device=self.device)
        self.class_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        self.confs = torch.empty((0,), device=self.device)
        if idx.numel():
            xyxy = _xywh_to_xyxy(out[:4, idx].T)
            keep = nms(xyxy, conf[idx], iou)[:max_det]
            if detection_index is not None:
                keep = keep[detection_index: detection_index + 1]
            sel = idx[keep]
            self.boxes, self.class_ids, self.confs = xyxy[keep], cls[sel], conf[sel]

    @property
    def has_targets(self) -> bool:
        return self.boxes.shape[0] > 0

    def targets(self) -> list[Detection]:
        names = self.model.info.class_names
        return [Detection(box=self.lb.box_to_original(b), class_id=int(c),
                          class_name=names[int(c)], conf=float(s))
                for b, c, s in zip(self.boxes.tolist(), self.class_ids.tolist(), self.confs.tolist())]

    # ---- timeout (cooperative) ----
    def set_timeout(self, seconds: float) -> None:
        self._timeout = seconds
        self._deadline = time.monotonic() + seconds

    def _check_deadline(self) -> None:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise XAITimeoutError(f"XAI method exceeded its time limit ({self._timeout:g}s)")

    # ---- the scalar target ----
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, S, S) in [0, 1]  ->  (B,)  (differentiable with respect to x)."""
        self._check_deadline()
        out = self._net(x)                                           # (B, 4+nc, A)
        with torch.no_grad():
            pred = _xywh_to_xyxy(out[:, :4, :].transpose(1, 2))      # (B, A, 4)
            overlap = _iou(self.boxes, pred) >= self.match_iou       # (B, K, A)
        scores = out[:, 4 + self.class_ids, :]                       # (B, K, A)
        scores = scores.masked_fill(~overlap, 0.0)
        return scores.max(dim=2).values.sum(dim=1)                   # (B,)

    @torch.no_grad()
    def score_numpy(self, batch: np.ndarray, chunk: int = 8) -> np.ndarray:
        """batch: (N, S, S, 3) with values 0-255 (uint8 or float)  ->  (N,) float64."""
        arr = np.asarray(batch, dtype=np.float32)
        outs = []
        for i in range(0, len(arr), chunk):
            t = torch.from_numpy(arr[i:i + chunk]).permute(0, 3, 1, 2).div(255.0).to(self.device)
            outs.append(self.score(t).detach().cpu().numpy())
        return np.concatenate(outs).astype(np.float64)