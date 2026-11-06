from pydantic import BaseModel


class XAIMethodRead(BaseModel):
    name: str
    display_name: str
    description: str
    applicable_architectures: list[str]
    supported_masks: list[str]
    avg_runtime_estimate: float | None     # seconds per image (ONLY the XAI), from the benchmark
    params_schema: dict
    slow: bool
    uses_target_layer: bool                # False: the frontend can hide the layer dropdown


class MaskRead(BaseModel):
    name: str
    description: str


class LayerRead(BaseModel):
    name: str
    description: str
    stride: int | None
    default: bool