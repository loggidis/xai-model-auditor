from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class SettingsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    batch_size: int
    notifications_enabled: bool
    theme: str


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int | None = Field(None, ge=1, le=50)
    notifications_enabled: bool | None = None
    theme: Literal["dark", "light"] | None = None