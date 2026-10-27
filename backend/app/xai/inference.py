from dataclasses import dataclass

import numpy as np

from app.xai.loader import LoadedModel


@dataclass
class Detection:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels της αρχικής εικόνας)
    class_id: int
    class_name: str
    conf: float

    def to_dict(self) -> dict:
        return {"box": [round(v, 1) for v in self.box], "class_id": self.class_id,
                "class_name": self.class_name, "conf": round(self.conf, 4)}


def detect(model: LoadedModel, image_rgb: np.ndarray, conf_threshold: float = 0.25,
           iou: float = 0.7, max_det: int = 300) -> list[Detection]:
    bgr = np.ascontiguousarray(image_rgb[..., ::-1])   # το Ultralytics περιμένει BGR για numpy
    res = model.yolo.predict(
        bgr, conf=conf_threshold, iou=iou, max_det=max_det,
        imgsz=model.info.input_size,
        device="cpu" if model.device.type == "cpu" else 0,
        verbose=False,
    )[0]
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    names = model.info.class_names
    return [Detection(tuple(float(v) for v in xyxy[i]), int(cls[i]), names[cls[i]], float(confs[i]))
            for i in range(len(cls))]