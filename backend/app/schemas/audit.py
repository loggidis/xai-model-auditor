import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class AuditCreate(BaseModel):
    model_id: uuid.UUID
    dataset_id: uuid.UUID
    xai_methods: list[str] = Field(min_length=1, max_length=10)
    target_layer: str | None = None          # None = default layer του μοντέλου
    mask: str | None = None                  # None = full_image
    conf_threshold: float = Field(0.25, ge=0.01, le=1.0)


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
    url: str | None                          # None αν δεν υπάρχει heatmap


class ResultDetail(BaseModel):
    image_id: uuid.UUID
    filename: str
    image_index: int
    batch_index: int
    status: str
    error_message: str | None
    detections: list[dict] | None
    heatmaps: list[HeatmapRead]