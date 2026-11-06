import cv2
import numpy as np

from app.xai.scalar_base import ScalarTargetMethod, timeout_param
from app.xai.target import TARGET_PARAMS


class ShapPartitionYOLO(ScalarTargetMethod):
    name = "shap_partition"
    display_name = "SHAP (Partition Explainer)"
    description = ("Shapley values with hierarchical partitioning of the image. Runs on a coarse grid "
                   "(grid_size), so the heatmap is more 'blocky'. Slow method.")
    params_schema = {
        "type": "object",
        "properties": {
            **TARGET_PARAMS,
            "max_evals": {"type": "integer", "minimum": 20, "maximum": 5000, "default": 300,
                          "description": "Number of model evaluations (more = better resolution)"},
            "grid_size": {"type": "integer", "minimum": 8, "maximum": 128, "default": 64,
                          "description": "Dimensions of the grid on which SHAP runs"},
            "masker": {"type": "string", "enum": ["blur", "inpaint_telea"], "default": "blur",
                       "description": "How regions are 'hidden'"},
            "blur_kernel": {"type": "integer", "minimum": 2, "maximum": 64, "default": 16,
                            "description": "Blur kernel (in grid pixels)"},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 32, "default": 8},
            "timeout_s": timeout_param(300),
        },
    }

    def attribute(self, adapter, p) -> np.ndarray:
        import shap                                           # lazy import (heavy)
        size, g = adapter.canvas.shape[0], p["grid_size"]
        canvas_f = adapter.canvas.astype(np.float32)
        low = cv2.resize(adapter.canvas, (g, g), interpolation=cv2.INTER_AREA).astype(np.float32)

        def f(batch_low: np.ndarray) -> np.ndarray:
            fulls = []
            for b in batch_low:
                hidden = (np.abs(b - low).max(axis=-1) > 1e-3).astype(np.uint8)    # (g, g)
                hidden_up = cv2.resize(hidden, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
                up = cv2.resize(b.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
                full = canvas_f.copy()
                full[hidden_up] = up[hidden_up]
                fulls.append(full)
            return adapter.score_numpy(np.stack(fulls), chunk=p["batch_size"])

        spec = f"blur({p['blur_kernel']},{p['blur_kernel']})" if p["masker"] == "blur" else "inpaint_telea"
        masker = shap.maskers.Image(spec, low.shape)
        explainer = shap.Explainer(f, masker, algorithm="partition")
        sv = explainer(low[None], max_evals=p["max_evals"], batch_size=p["batch_size"], silent=True)

        v = np.asarray(sv.values)[0]                          # (g, g, 3) or (g, g, 3, outputs)
        while v.ndim > 3:
            v = v.sum(axis=-1)
        a_low = np.clip(v, 0, None).sum(axis=-1).astype(np.float32)     # positive contribution only
        return cv2.resize(a_low, (size, size), interpolation=cv2.INTER_LINEAR)