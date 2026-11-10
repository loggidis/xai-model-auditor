import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: Literal["en", "el"] = "en"
    include_filenames: bool = False          # file names are sent to the LLM provider only if true
    top_flagged: int = Field(10, ge=0, le=25)


class SummaryRead(BaseModel):
    audit_id: uuid.UUID
    status: str                              # not_requested | pending | running | done | failed
    summary: str | None
    model: str | None
    params: dict | None
    error_message: str | None
    requested_at: datetime | None
    finished_at: datetime | None