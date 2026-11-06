import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from app.xai.tagging import TaggingConfig

class AuditCreate(BaseModel):
    model_id: uuid.UUID
    dataset_id: uuid.UUID
    xai_methods: list[str] = Field(min_length=1, max_length=10)
    target_layer: str | None = None          # None = the model's default layer
    mask: str | None = None                  # None = full_image (heatmap region)
    conf_threshold: float = Field(0.25, ge=0.01, le=1.0)
    method_params: dict[str, dict] = Field(default_factory=dict)   # e.g. {"lime_image": {"num_samples": 500}}
    tagging: TaggingConfig | None = None     # None = defaults


class Progress(BaseModel):
    total: int
    done: int
    failed: int
    processing: int
    pending: int
    percent: float                           # (done + failed) / total


class AuditRead(BaseModel):
    id: uuid.UUID
    model_id: uuid.UUID
    dataset_id: uuid.UUID
    status: str
    config: dict
    progress: Progress
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class HeatmapBrief(BaseModel):
    method: str
    status: str


class ResultSummary(BaseModel):
    image_id: uuid.UUID
    filename: str
    image_index: int
    batch_index: int
    status: str
    detection_count: int
    agreement_score: float | None
    auto_tag: str | None
    heatmaps: list[HeatmapBrief]


class ResultPage(BaseModel):
    items: list[ResultSummary]
    total: int
    limit: int
    offset: int


class HeatmapRead(BaseModel):
    method: str
    status: str
    runtime_ms: int | None
    error_message: str | None
    box_focus: float | None
    url: str | None                          # None if no heatmap


class ResultDetail(BaseModel):
    image_id: uuid.UUID
    filename: str
    image_index: int
    batch_index: int
    status: str
    error_message: str | None
    detections: list[dict] | None
    agreement_score: float | None
    auto_tag: str | None
    auto_tag_reason: str | None
    agreement_details: dict | None           # pairwise IoU/Pearson and aggregate
    heatmaps: list[HeatmapRead]