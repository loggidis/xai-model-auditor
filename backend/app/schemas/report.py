import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class PieSlice(BaseModel):
    key: str
    count: int
    percent: float


class ClassStats(BaseModel):
    class_name: str
    detections: int
    images: int                          # images that contain at least one detection of this class
    avg_confidence: float | None         # mean model confidence over the detections (0-1)
    evaluated_images: int                # evaluated images whose primary class is this one
    avg_trust: float | None              # mean user trust (0-100) over those images


class AuditReport(BaseModel):
    audit_id: uuid.UUID
    status: str
    total_images: int
    processed_images: int                # status == done
    failed_images: int
    remaining_images: int                # pending + processing
    evaluated_images: int
    by_auto_tag: list[PieSlice]
    by_category: list[PieSlice]          # includes "unevaluated"
    avg_confidence: float | None         # model confidence over all detections
    avg_trust: float | None              # user trust over all evaluations
    avg_agreement: float | None
    classes: list[ClassStats]


class FlaggedItem(BaseModel):
    image_id: uuid.UUID
    filename: str
    image_index: int
    reason: str                          # user:wrong_prediction | user:correct_wrong_reason | user:low_trust | auto:needs_review
    score_source: str                    # evaluation | agreement | none
    sort_score: float | None             # 0-100, ascending order (trust, or agreement * 100 as fallback)
    category: str | None
    trust_score: int | None
    notes: str | None
    auto_tag: str | None
    agreement_score: float | None
    mean_confidence: float | None
    detection_count: int


class FlaggedPage(BaseModel):
    items: list[FlaggedItem]
    total: int
    limit: int
    offset: int


class AuditReportRow(BaseModel):
    audit_id: uuid.UUID
    created_at: datetime
    status: str
    model_id: uuid.UUID
    model_name: str
    dataset_id: uuid.UUID
    dataset_name: str
    total_images: int
    processed_images: int
    failed_images: int
    evaluated_images: int
    needs_review_images: int             # auto_tag == needs_review
    problematic_evaluations: int         # category != correct_right_reason
    avg_confidence: float | None
    avg_trust: float | None
    avg_agreement: float | None


class GlobalReport(BaseModel):
    audits_total: int
    audits_by_status: list[PieSlice]
    images_processed: int
    evaluated_images: int
    by_auto_tag: list[PieSlice]
    by_category: list[PieSlice]
    avg_confidence: float | None
    avg_trust: float | None
    avg_agreement: float | None
    classes: list[ClassStats]            # aggregated by class name across all audits
    audits: list[AuditReportRow]         # paginated, newest first
    limit: int
    offset: int


class ExportRequest(BaseModel):
    trust_below: int = Field(50, ge=0, le=100)        # 0 disables the low-trust rule
    include_unevaluated: bool = True                   # include auto-flagged (needs_review) images
    include_predictions: bool = False                  # add model predictions as YOLO txt files


class ExportRead(BaseModel):
    id: uuid.UUID
    audit_id: uuid.UUID
    status: str
    params: dict
    image_count: int
    skipped_count: int
    file_size: int | None
    error_message: str | None
    created_at: datetime
    finished_at: datetime | None
    download_url: str | None