import time

import numpy as np
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from torchvision.ops import nms

from app.xai.base import BaseXAIMethod, XAIResult
from app.xai.inference import Detection
from app.xai.loader import LoadedModel, UnsupportedModelError
from app.xai.masks import MASKS, boxes_region
from app.xai.preprocess import letterbox, to_tensor


class _TensorOutput(torch.nn.Module):
    """Ultralytics returns a tuple. pytorch-grad-cam expects a single tensor of shape (B, 4+nc, anchors)."""

    def __init__(self, net: torch.nn.Module, num_classes: int):
        super().__init__()
        self.net = net
        self.expected = 4 + num_classes

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not (torch.is_tensor(out) and out.ndim == 3 and out.shape[1] == self.expected):
            raise UnsupportedModelError(
                "Unexpected output layout (expected (B, 4+nc, anchors)). "
                "End-to-end/NMS-free models are not supported yet."
            )
        return out


class DetectionScoreTarget:
    """Scalar target: sum of class scores of the selected detections.
    The selection (threshold + NMS) is done here, in the same forward pass."""

    def __init__(self, conf: float, iou: float, max_det: int, detection_index: int | None):
        self.conf, self.iou, self.max_det, self.index = conf, iou, max_det, detection_index
        self.boxes: list[list[float]] = []     # xyxy in letterbox coordinates  
        self.class_ids: list[int] = []
        self.confs: list[float] = []

    def __call__(self, out: torch.Tensor) -> torch.Tensor:   # out: (4+nc, anchors)
        self.boxes, self.class_ids, self.confs = [], [], []
        with torch.no_grad():
            conf, cls = out[4:].max(dim=0)
            idx = (conf >= self.conf).nonzero(as_tuple=True)[0]
            sel = idx[:0]
            if idx.numel():
                xywh = out[:4, idx].T
                xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], dim=1)
                keep = nms(xyxy, conf[idx], self.iou)[: self.max_det]   # sorted by descending conf
                if self.index is not None:
                    keep = keep[self.index: self.index + 1]
                sel = idx[keep]
                self.boxes = xyxy[keep].tolist()
                self.class_ids = cls[sel].tolist()
                self.confs = conf[sel].tolist()
        if sel.numel() == 0:
            return out.sum() * 0          # no detections -> zero heatmap
        return out[4 + cls[sel], sel].sum()


class GradCAMPlusPlusYOLO(BaseXAIMethod):
    name = "gradcam_pp"
    display_name = "Grad-CAM++"
    description = ("Gradient-based heatmap overlay on a network feature map."
                   "Target: detections above the confidence threshold.")
    applicable_architectures = ["yolo_detect"]
    supported_masks = ["full_image", "detection_boxes", "background"]
    params_schema = {
        "type": "object",
        "properties": {
            "iou": {"type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.7,
                    "description": "NMS IoU for the selection of target detections"},
            "max_det": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20,
                        "description": "Maximum number of detections in the target"},
            "detection_index": {"type": ["integer", "null"], "minimum": 0, "default": None,
                                "description": "null = all detections, k = only the k-th (by descending conf)"},
        },
    }

    def run(self, model: LoadedModel, image: np.ndarray, target_layer: str | None = None,
            mask: str | None = None, conf_threshold: float = 0.25, **params) -> XAIResult:
        p = self.resolve_params(params)
        mask_name = mask or "full_image"
        if mask_name not in self.supported_masks:
            raise ValueError(f"Unsupported mask '{mask_name}'")
        layer_name = target_layer or model.default_layer()
        layer = model.get_layer(layer_name)

        t0 = time.perf_counter()
        canvas, lb = letterbox(image, model.info.input_size)
        # requires_grad on the input: the params might be frozen (after the fuse in Ultralytics)
        x = to_tensor(canvas, model.device).requires_grad_(True)
        target = DetectionScoreTarget(conf_threshold, p["iou"], p["max_det"], p["detection_index"])
        wrapped = _TensorOutput(model.net, model.info.num_classes)

        with torch.enable_grad():
            with GradCAMPlusPlus(model=wrapped, target_layers=[layer]) as cam:
                cam_map = cam(input_tensor=x, targets=[target])[0]

        heat = lb.heatmap_to_original(cam_map)
        targets = [
            Detection(box=lb.box_to_original(b), class_id=int(c),
                      class_name=model.info.class_names[int(c)], conf=float(s))
            for b, c, s in zip(target.boxes, target.class_ids, target.confs)
        ]
        boxes = [d.box for d in targets]
        total = float(heat.sum())
        box_focus = (float(heat[boxes_region(heat.shape, boxes)].sum() / total)
                     if targets and total > 0 else None)
        heat = MASKS[mask_name].apply(heat, boxes)

        if model.device.type == "cuda":
            torch.cuda.synchronize()
        return XAIResult(heatmap=heat, method=self.name, layer=layer_name, mask=mask_name,
                         targets=targets, box_focus=box_focus,
                         runtime_s=time.perf_counter() - t0)