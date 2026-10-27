from pydantic import BaseModel


class XAIMethodRead(BaseModel):
    name: str
    display_name: str
    description: str
    applicable_architectures: list[str]
    supported_masks: list[str]
    avg_runtime_estimate: float | None     # δευτερόλεπτα ανά εικόνα (ΜΟΝΟ το XAI)
    params_schema: dict


class MaskRead(BaseModel):
    name: str
    description: str


class LayerRead(BaseModel):
    name: str
    description: str
    stride: int | None
    default: bool