import time

import numpy as np
import torch

from app.xai.base import BaseXAIMethod, XAIResult
from app.xai.masks import MASKS, boxes_region
from app.xai.target import DetectionTargetAdapter

REGION_MASKS = ["full_image", "detection_boxes", "background"]


def timeout_param(default: float) -> dict:
    return {"type": "number", "minimum": 1, "maximum": 3600, "default": default,
            "description": "Maximum time in seconds per image. If exceeded, the method fails (not the image)"}


def normalize(a: np.ndarray) -> np.ndarray:
    """Keeps the positive contribution and scales to [0,1] (99th percentile of the positive values)."""
    a = np.clip(np.nan_to_num(a.astype(np.float32)), 0, None)
    pos = a[a > 0]
    if pos.size == 0:
        return np.zeros_like(a)
    hi = float(np.percentile(pos, 99.0))
    return np.clip(a / hi, 0, 1) if hi > 0 else np.zeros_like(a)


class ScalarTargetMethod(BaseXAIMethod):
    applicable_architectures = ["yolo_detect"]
    supported_masks = REGION_MASKS
    uses_target_layer = False
    slow = True

    def attribute(self, adapter: DetectionTargetAdapter, p: dict) -> np.ndarray:
        """Returns an (S, S) map >= 0 on the letterbox canvas."""
        raise NotImplementedError

    def run(self, model, image, target_layer=None, mask=None, conf_threshold=0.25, **params) -> XAIResult:
        p = self.resolve_params(params)
        mask_name = mask or "full_image"
        if mask_name not in self.supported_masks:
            raise ValueError(f"Unsupported mask '{mask_name}'")

        t0 = time.perf_counter()
        adapter = DetectionTargetAdapter(model, image, conf_threshold, iou=p["iou"],
                                         max_det=p["max_det"], detection_index=p["detection_index"])
        targets = adapter.targets()
        if targets:
            adapter.set_timeout(p["timeout_s"])
            with torch.enable_grad():
                canvas_map = self.attribute(adapter, p)
            heat = adapter.lb.heatmap_to_original(normalize(canvas_map))
        else:
            heat = np.zeros(image.shape[:2], dtype=np.float32)       # no detection -> empty heatmap

        boxes = [d.box for d in targets]
        total = float(heat.sum())
        box_focus = (float(heat[boxes_region(heat.shape, boxes)].sum() / total)
                     if targets and total > 0 else None)
        heat = MASKS[mask_name].apply(heat, boxes)

        if model.device.type == "cuda":
            torch.cuda.synchronize()
        return XAIResult(heatmap=heat, method=self.name, layer="input", mask=mask_name,
                         targets=targets, box_focus=box_focus, runtime_s=time.perf_counter() - t0)