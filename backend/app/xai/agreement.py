import itertools

import cv2
import numpy as np


def _prep(heatmap: np.ndarray, size: int) -> np.ndarray:
    small = cv2.resize(np.asarray(heatmap, dtype=np.float32), (size, size), interpolation=cv2.INTER_AREA)
    mx = float(small.max())
    return small / mx if mx > 0 else small


def iou_thresholded(a: np.ndarray, b: np.ndarray, thr: float) -> float | None:
    A, B = a >= thr, b >= thr
    union = int((A | B).sum())
    return None if union == 0 else float((A & B).sum() / union)    # both empty -> undefined


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return None if den < 1e-12 else float((x * y).sum() / den)      # constant map -> undefined


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def compute_agreement(maps: dict[str, np.ndarray], iou_threshold: float = 0.5,
                      iou_weight: float = 0.5, size: int = 128) -> dict | None:
    """Pairwise IoU/Pearson and aggregate score (or None if there are < 2 heatmaps)."""
    names = sorted(maps)
    if len(names) < 2:
        return None
    prepped = {n: _prep(maps[n], size) for n in names}
    pairs = []
    for a, b in itertools.combinations(names, 2):
        pairs.append({"methods": [a, b],
                      "iou": _r(iou_thresholded(prepped[a], prepped[b], iou_threshold)),
                      "pearson": _r(pearson(prepped[a], prepped[b]))})
    iou_mean = _mean([p["iou"] for p in pairs if p["iou"] is not None])
    pr_mean = _mean([p["pearson"] for p in pairs if p["pearson"] is not None])

    parts = []
    if iou_mean is not None:
        parts.append((iou_weight, iou_mean))
    if pr_mean is not None:
        parts.append((1 - iou_weight, max(pr_mean, 0.0)))       # negative correlation = 0
    wsum = sum(w for w, _ in parts)
    score = sum(w * v for w, v in parts) / wsum if wsum > 0 else None
    return {"score": _r(score), "iou_mean": _r(iou_mean), "pearson_mean": _r(pr_mean),
            "iou_threshold": iou_threshold, "iou_weight": iou_weight,
            "methods": names, "pairs": pairs}