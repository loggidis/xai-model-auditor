from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class Letterbox:
    """Μετασχηματισμός original -> τετράγωνο input του μοντέλου (και αντίστροφα)."""
    scale: float
    left: int
    top: int
    new_w: int
    new_h: int
    orig_w: int
    orig_h: int

    def heatmap_to_original(self, hm: np.ndarray) -> np.ndarray:
        crop = hm[self.top:self.top + self.new_h, self.left:self.left + self.new_w]
        out = cv2.resize(crop, (self.orig_w, self.orig_h), interpolation=cv2.INTER_LINEAR)
        return np.nan_to_num(out).astype(np.float32)

    def box_to_original(self, xyxy) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        x1, x2 = [min(max((v - self.left) / self.scale, 0.0), self.orig_w) for v in (x1, x2)]
        y1, y2 = [min(max((v - self.top) / self.scale, 0.0), self.orig_h) for v in (y1, y2)]
        return (x1, y1, x2, y2)


def letterbox(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    h, w = image_rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, Letterbox(r, left, top, nw, nh, w, h)


def to_tensor(canvas_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(canvas_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return t.to(device)