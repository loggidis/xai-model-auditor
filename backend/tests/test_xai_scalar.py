import numpy as np
import pytest
import torch

pytest.importorskip("ultralytics")
pytest.importorskip("captum")
pytest.importorskip("shap")
pytest.importorskip("lime")
from ultralytics import YOLO

from app.xai.loader import from_yolo
from app.xai.registry import get_method, list_methods
from app.xai.target import DetectionTargetAdapter, XAITimeoutError

FAST = {
    "integrated_gradients": {"n_steps": 4, "batch_size": 2, "smooth_sigma": 2.0},
    "lime_image": {"num_samples": 20, "n_segments": 10, "batch_size": 4},
    "shap_partition": {"max_evals": 30, "grid_size": 16, "batch_size": 4, "blur_kernel": 4},
}


@pytest.fixture(scope="module")
def tiny():
    return from_yolo(YOLO("yolov8n.yaml"))


def _img():
    return np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)


def test_registry_has_all_methods():
    names = {m.name for m in list_methods("yolo_detect")}
    assert {"gradcam_pp", *FAST} <= names


def test_adapter_score_is_batched_scalar(tiny):
    ad = DetectionTargetAdapter(tiny, _img(), 0.0, max_det=3)
    assert ad.has_targets and 1 <= len(ad.targets()) <= 3
    t = torch.from_numpy(ad.canvas).permute(2, 0, 1).float().div(255).unsqueeze(0)
    s = ad.score(torch.cat([t, t]))
    assert s.shape == (2,) and torch.isfinite(s).all()


def test_adapter_timeout(tiny):
    ad = DetectionTargetAdapter(tiny, _img(), 0.0, max_det=3)
    ad.set_timeout(0)
    with pytest.raises(XAITimeoutError):
        ad.score_numpy(ad.canvas[None])


@pytest.mark.parametrize("name", list(FAST))
def test_method_heatmap(tiny, name):
    res = get_method(name).run(tiny, _img(), None, "full_image",
                               conf_threshold=0.0, max_det=3, **FAST[name])
    assert res.heatmap.shape == (300, 400)
    assert np.isfinite(res.heatmap).all()
    assert 0.0 <= res.heatmap.min() and res.heatmap.max() <= 1.0 + 1e-6
    assert 1 <= len(res.targets) <= 3 and res.layer == "input"


def test_no_target_gives_empty_heatmap(tiny):
    res = get_method("integrated_gradients").run(tiny, _img(), None, None,
                                                 conf_threshold=0.999999, **FAST["integrated_gradients"])
    assert res.targets == [] and res.heatmap.max() == 0 and res.box_focus is None


def test_param_validation(tiny):
    m = get_method("integrated_gradients")
    with pytest.raises(ValueError):
        m.run(tiny, _img(), None, None, n_steps=0)
    with pytest.raises(ValueError):
        m.run(tiny, _img(), None, None, baseline="red")
    with pytest.raises(ValueError):
        m.run(tiny, _img(), None, None, foo=1)