import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from app.xai.inference import Detection
from app.xai.loader import LoadedModel

RUNTIME_FILE = Path(__file__).with_name("runtime_estimates.json")


def load_runtime_estimates() -> dict:
    try:
        return json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_runtime_estimate(method: str, seconds: float, device: str) -> None:
    data = load_runtime_estimates()
    data[method] = {"seconds": round(seconds, 3), "device": device}
    RUNTIME_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass
class XAIResult:
    heatmap: np.ndarray            # (H, W) float32 στο [0, 1], ανάλυση αρχικής εικόνας
    method: str
    layer: str
    mask: str
    targets: list[Detection]       # οι detections που εξηγήθηκαν
    box_focus: float | None        # ποσοστό θερμότητας μέσα στα target boxes (πριν το mask)
    runtime_s: float


class BaseXAIMethod(ABC):
    name: ClassVar[str]
    display_name: ClassVar[str]
    description: ClassVar[str]
    params_schema: ClassVar[dict]                  # JSON Schema
    applicable_architectures: ClassVar[list[str]]
    supported_masks: ClassVar[list[str]]

    @property
    def avg_runtime_estimate(self) -> float | None:
        entry = load_runtime_estimates().get(self.name)
        return entry["seconds"] if entry else None

    def resolve_params(self, params: dict) -> dict:
        props = self.params_schema.get("properties", {})
        unknown = set(params) - set(props)
        if unknown:
            raise ValueError(f"Unknown params for {self.name}: {sorted(unknown)}")
        return {k: params.get(k, v.get("default")) for k, v in props.items()}

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "applicable_architectures": self.applicable_architectures,
            "supported_masks": self.supported_masks,
            "avg_runtime_estimate": self.avg_runtime_estimate,
            "params_schema": self.params_schema,
        }

    @abstractmethod
    def run(self, model: LoadedModel, image: np.ndarray, target_layer: str | None,
            mask: str | None, conf_threshold: float = 0.25, **params) -> XAIResult:
        """image: RGB uint8 (H, W, 3)."""