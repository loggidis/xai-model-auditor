import numpy as np
import pytest

from app.xai.agreement import compute_agreement
from app.xai.tagging import TaggingConfig, decide_tag


def _blob(x0, x1):
    m = np.zeros((200, 200), dtype=np.float32)
    m[50:150, x0:x1] = 1.0
    return m


def test_identical_maps_agree_fully():
    a = compute_agreement({"a": _blob(20, 100), "b": _blob(20, 100)})
    assert a["score"] == pytest.approx(1.0, abs=1e-3)
    assert a["pairs"][0]["iou"] == pytest.approx(1.0)


def test_disjoint_maps_disagree():
    a = compute_agreement({"a": _blob(10, 60), "b": _blob(120, 180)})
    assert a["pairs"][0]["iou"] == 0.0
    assert a["score"] == 0.0                      # negative/zero Pearson -> 0


def test_constant_map_has_no_pearson():
    a = compute_agreement({"a": np.ones((50, 50), np.float32), "b": _blob(20, 100)})
    assert a["pairs"][0]["pearson"] is None


def test_single_map_has_no_agreement():
    assert compute_agreement({"a": _blob(20, 100)}) is None


def test_pairs_for_three_methods():
    a = compute_agreement({"a": _blob(20, 100), "b": _blob(20, 100), "c": _blob(30, 110)})
    assert len(a["pairs"]) == 3


def test_tag_rules():
    cfg = TaggingConfig()
    kw = dict(mean_conf=0.9, min_box_focus=0.9)
    assert decide_tag(cfg, has_targets=False, agreement=None, **kw)[0] == "no_detection"
    assert decide_tag(cfg, has_targets=True, agreement=None, **kw)[0] == "inconclusive"
    assert decide_tag(cfg, has_targets=True, agreement=0.1, **kw)[0] == "needs_review"
    assert decide_tag(cfg, has_targets=True, agreement=0.8, **kw)[0] == "consistent"
    assert decide_tag(cfg, has_targets=True, agreement=0.35, **kw)[0] == "uncertain"
    assert decide_tag(cfg, has_targets=True, agreement=0.8, mean_conf=0.3, min_box_focus=0.9)[0] == "uncertain"
    assert decide_tag(cfg, has_targets=True, agreement=0.8, mean_conf=0.9, min_box_focus=0.05)[0] == "needs_review"


def test_box_focus_rule_can_be_disabled():
    cfg = TaggingConfig(box_focus_min=None)
    assert decide_tag(cfg, has_targets=True, agreement=0.8, mean_conf=0.9, min_box_focus=0.0)[0] == "consistent"


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        TaggingConfig(agreement_low=0.9, agreement_high=0.2)