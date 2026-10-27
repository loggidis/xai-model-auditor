from abc import ABC, abstractmethod

import numpy as np


def boxes_region(shape: tuple[int, int], boxes) -> np.ndarray:
    """Boolean μάσκα (H, W): True μέσα σε οποιοδήποτε box."""
    h, w = shape
    region = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        region[max(int(y1), 0):min(int(np.ceil(y2)), h),
               max(int(x1), 0):min(int(np.ceil(x2)), w)] = True
    return region


class BaseMask(ABC):
    name: str
    description: str

    @abstractmethod
    def apply(self, heatmap: np.ndarray, boxes: list) -> np.ndarray: ...

    def metadata(self) -> dict:
        return {"name": self.name, "description": self.description}


class FullImageMask(BaseMask):
    name = "full_image"
    description = "Ολόκληρο το heatmap, χωρίς περιορισμό."

    def apply(self, heatmap, boxes):
        return heatmap


class DetectionBoxesMask(BaseMask):
    name = "detection_boxes"
    description = "Μόνο η θερμότητα μέσα στα bounding boxes των detections."

    def apply(self, heatmap, boxes):
        return heatmap * boxes_region(heatmap.shape, boxes)


class BackgroundMask(BaseMask):
    name = "background"
    description = "Μόνο η θερμότητα έξω από τα boxes (πού κοιτάζει το μοντέλο στο φόντο)."

    def apply(self, heatmap, boxes):
        return heatmap * ~boxes_region(heatmap.shape, boxes)


MASKS: dict[str, BaseMask] = {m.name: m for m in (FullImageMask(), DetectionBoxesMask(), BackgroundMask())}