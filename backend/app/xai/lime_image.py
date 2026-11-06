import numpy as np

from app.xai.scalar_base import ScalarTargetMethod, timeout_param
from app.xai.target import TARGET_PARAMS

_HIDE = {"mean": None, "gray": 114, "black": 0}


class LimeImageYOLO(ScalarTargetMethod):
    name = "lime_image"
    display_name = "LIME"
    description = ("Splits the image into superpixels, randomly hides subsets of them and fits a "
                   "linear model to the score of the detections. Slow method.")
    params_schema = {
        "type": "object",
        "properties": {
            **TARGET_PARAMS,
            "num_samples": {"type": "integer", "minimum": 10, "maximum": 5000, "default": 300,
                            "description": "Number of random variations of the image"},
            "n_segments": {"type": "integer", "minimum": 5, "maximum": 500, "default": 80,
                           "description": "Approximate number of superpixels (SLIC)"},
            "masker": {"type": "string", "enum": ["mean", "gray", "black"], "default": "mean",
                       "description": "What a hidden superpixel is replaced with"},
            "seed": {"type": "integer", "minimum": 0, "maximum": 2**31 - 1, "default": 0},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 32, "default": 8},
            "timeout_s": timeout_param(180),
        },
    }

    def attribute(self, adapter, p) -> np.ndarray:
        from lime import lime_image                           # lazy imports
        from skimage.segmentation import slic

        def segmentation_fn(img):
            return slic(img, n_segments=p["n_segments"], compactness=10, start_label=0)

        explainer = lime_image.LimeImageExplainer(random_state=p["seed"])
        exp = explainer.explain_instance(
            adapter.canvas,
            lambda batch: adapter.score_numpy(batch, chunk=p["batch_size"])[:, None],   # shape (N, 1)
            labels=(0,), top_labels=None, hide_color=_HIDE[p["masker"]],
            num_samples=p["num_samples"], batch_size=p["batch_size"],
            segmentation_fn=segmentation_fn, random_seed=p["seed"],
        )
        heat = np.zeros(exp.segments.shape, dtype=np.float32)
        for seg_id, weight in exp.local_exp[0]:
            if weight > 0:                                    # only what supports the score
                heat[exp.segments == seg_id] = weight
        return heat