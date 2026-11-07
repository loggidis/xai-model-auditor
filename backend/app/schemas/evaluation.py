import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvaluationCategory = Literal["correct_right_reason", "correct_wrong_reason", "wrong_prediction"]


class EvaluationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: uuid.UUID
    trust_score: int = Field(ge=0, le=100)
    category: EvaluationCategory
    notes: str | None = Field(None, max_length=2000)


class EvaluationUpdate(BaseModel):
    """Partial update: only the fields that are sent are changed. notes=null clears the notes."""
    model_config = ConfigDict(extra="forbid")
    trust_score: int | None = Field(None, ge=0, le=100)
    category: EvaluationCategory | None = None
    notes: str | None = Field(None, max_length=2000)


class EvaluationRead(BaseModel):
    id: uuid.UUID
    audit_id: uuid.UUID
    image_id: uuid.UUID
    filename: str
    trust_score: int
    category: str
    notes: str | None
    created_at: datetime
    updated_at: datetime