import numpy as np
import pytest

pytest.importorskip("ultralytics")
from ultralytics import YOLO

from app.xai.loader import from_yolo
from app.xai.masks import boxes_region
from app.xai.registry import get_method


@pytest.fixture(scope="module")
def tiny():
    return from_yolo(YOLO("yolov8n.yaml"))


def _img():
    return np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)


def test_recommended_layers(tiny):
    layers = tiny.info.recommended_layers
    assert len(layers) == 3
    assert sum(l.default for l in layers) == 1


def test_gradcam_heatmap(tiny):
    # conf=0 -> επιλέγονται σίγουρα anchors, ακόμα και με τυχαία βάρη
    res = get_method("gradcam_pp").run(tiny, _img(), None, "full_image", conf_threshold=0.0, max_det=5)
    assert res.heatmap.shape == (300, 400)
    assert np.isfinite(res.heatmap).all()
    assert 0.0 <= res.heatmap.min() and res.heatmap.max() <= 1.0 + 1e-6
    assert 1 <= len(res.targets) <= 5


def test_background_mask_is_zero_inside_boxes(tiny):
    res = get_method("gradcam_pp").run(tiny, _img(), None, "background", conf_threshold=0.0, max_det=3)
    region = boxes_region(res.heatmap.shape, [d.box for d in res.targets])
    assert res.heatmap[region].max() == 0


def test_unknown_param_rejected(tiny):
    with pytest.raises(ValueError):
        get_method("gradcam_pp").run(tiny, _img(), None, None, foo=1)