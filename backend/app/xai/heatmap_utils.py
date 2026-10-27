import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.services import storage

_METHOD_RE = re.compile(r"[a-z0-9_]+")


def read_image_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png(path: Path, array: np.ndarray) -> None:
    """array: HxW (gray) ή HxWx3 (RGB), uint8."""
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", array)
    if not ok:
        raise ValueError("PNG encoding failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def colorize(heatmap: np.ndarray) -> np.ndarray:
    u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Η διαφάνεια ακολουθεί την ένταση: όπου δεν υπάρχει θερμότητα, φαίνεται η εικόνα."""
    a = np.clip(heatmap, 0, 1)[..., None] * alpha
    out = image_rgb.astype(np.float32) * (1 - a) + colorize(heatmap).astype(np.float32) * a
    return out.astype(np.uint8)


def draw_detections(image_rgb: np.ndarray, detections) -> np.ndarray:
    out = image_rgb.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(out, f"{d.class_name} {d.conf:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def heatmap_rel_path(audit_id: uuid.UUID, image_id: uuid.UUID, method: str, raw: bool = False) -> str:
    """Σταθερό naming: audits/{audit_id}/{image_id}/{method}.png (σχετικό με το storage)."""
    if not _METHOD_RE.fullmatch(method):
        raise ValueError("Invalid method name")
    suffix = ".raw.png" if raw else ".png"
    return f"audits/{audit_id}/{image_id}/{method}{suffix}"


def save_heatmap_files(audit_id: uuid.UUID, image_id: uuid.UUID, method: str,
                       image_rgb: np.ndarray, heatmap: np.ndarray) -> str:
    """Γράφει {method}.png (overlay) και {method}.raw.png (8-bit grayscale). Επιστρέφει το σχετικό path του overlay."""
    rel = heatmap_rel_path(audit_id, image_id, method)
    write_png(storage.abs_path(rel), overlay(image_rgb, heatmap))
    raw_rel = heatmap_rel_path(audit_id, image_id, method, raw=True)
    write_png(storage.abs_path(raw_rel), (np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
    return rel