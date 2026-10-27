import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.core.config import settings


class UnsupportedModelError(Exception):
    """Το αρχείο δεν είναι μοντέλο που υποστηρίζουμε."""


@dataclass
class LayerInfo:
    name: str
    description: str
    stride: int | None
    default: bool = False


@dataclass
class ModelInfo:
    architecture_type: str          # "yolo_detect"
    task: str                       # "detection"
    num_classes: int
    input_size: int
    class_names: list[str]
    recommended_layers: list[LayerInfo]


@dataclass
class LoadedModel:
    yolo: Any                       # ultralytics.YOLO
    net: torch.nn.Module            # yolo.model (το "γυμνό" PyTorch module)
    info: ModelInfo
    device: torch.device

    def default_layer(self) -> str:
        return next(l.name for l in self.info.recommended_layers if l.default)

    def get_layer(self, name: str) -> torch.nn.Module:
        modules = dict(self.net.named_modules())
        if not name or name not in modules:
            raise ValueError(f"Unknown layer '{name}'")
        return modules[name]


def get_device() -> torch.device:
    pref = settings.XAI_DEVICE
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _recommended_layers(net: torch.nn.Module, imgsz: int, device: torch.device) -> list[LayerInfo]:
    head = net.model[-1]
    if not hasattr(head, "f"):
        raise UnsupportedModelError("Unsupported detection head (no feature inputs found)")
    idxs = list(head.f)
    sizes: dict[int, int] = {}
    hooks = [
        net.model[i].register_forward_hook(lambda m, inp, out, k=i: sizes.__setitem__(k, out.shape[-1]))
        for i in idxs
    ]
    try:
        with torch.no_grad():
            net(torch.zeros(1, 3, imgsz, imgsz, device=device))
    finally:
        for h in hooks:
            h.remove()

    layers = []
    for i in idxs:
        stride = imgsz // sizes[i]
        p = int(math.log2(stride))
        layers.append(LayerInfo(
            name=f"model.{i}",
            description=f"Neck output P{p} (stride {stride}, {sizes[i]}x{sizes[i]} feature map)",
            stride=stride,
        ))
    default = next((j for j, l in enumerate(layers) if l.stride == 16), len(layers) // 2)
    layers[default].default = True
    return layers


def from_yolo(yolo) -> LoadedModel:
    if yolo.task != "detect":
        raise UnsupportedModelError(f"Unsupported task '{yolo.task}' (only detection is supported)")
    device = get_device()
    net = yolo.model
    if not isinstance(net, torch.nn.Module):
        raise UnsupportedModelError("Unsupported model object")
    net = net.to(device).float().eval()

    names = yolo.names
    class_names = [str(names[i]) for i in sorted(names)]
    imgsz = yolo.overrides.get("imgsz") or 640
    if isinstance(imgsz, (list, tuple)):
        imgsz = imgsz[0]
    imgsz = int(imgsz)

    info = ModelInfo(
        architecture_type="yolo_detect",
        task="detection",
        num_classes=len(class_names),
        input_size=imgsz,
        class_names=class_names,
        recommended_layers=_recommended_layers(net, imgsz, device),
    )
    return LoadedModel(yolo=yolo, net=net, info=info, device=device)


def load_model(path: Path) -> LoadedModel:
    from ultralytics import YOLO   # lazy import (βαρύ)
    try:
        yolo = YOLO(str(path))
    except Exception as e:
        raise UnsupportedModelError(f"Cannot load model: {e}") from e
    return from_yolo(yolo)