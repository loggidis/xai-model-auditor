import cv2
import numpy as np
import torch

from app.xai.preprocess import to_tensor
from app.xai.scalar_base import ScalarTargetMethod, timeout_param
from app.xai.target import TARGET_PARAMS


class IntegratedGradientsYOLO(ScalarTargetMethod):
    name = "integrated_gradients"
    display_name = "Integrated Gradients"
    description = ("Attributes the score of the detections to the image pixels, by integrating gradients "
                   "from a reference image (baseline) to the actual one. Independent of layer.")
    params_schema = {
        "type": "object",
        "properties": {
            **TARGET_PARAMS,
            "n_steps": {"type": "integer", "minimum": 4, "maximum": 200, "default": 16,
                        "description": "Integration steps (more = more accurate, slower)"},
            "baseline": {"type": "string", "enum": ["gray", "black"], "default": "gray",
                         "description": "Reference image"},
            "smooth_sigma": {"type": "number", "minimum": 0, "maximum": 20, "default": 4.0,
                             "description": "Gaussian blur (pixels) over the noisy map"},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 32, "default": 4,
                           "description": "Internal batch (memory)"},
            "timeout_s": timeout_param(120),
        },
    }

    def attribute(self, adapter, p) -> np.ndarray:
        from captum.attr import IntegratedGradients          # lazy import
        x = to_tensor(adapter.canvas, adapter.device)
        baseline = torch.full_like(x, 114 / 255) if p["baseline"] == "gray" else torch.zeros_like(x)
        ig = IntegratedGradients(adapter.score)
        attr = ig.attribute(x, baselines=baseline, n_steps=p["n_steps"],
                            internal_batch_size=p["batch_size"])
        a = attr.detach().abs().sum(dim=1)[0].cpu().numpy()   # |attribution| summed over channels
        if p["smooth_sigma"] > 0:
            a = cv2.GaussianBlur(a, (0, 0), p["smooth_sigma"])
        return a