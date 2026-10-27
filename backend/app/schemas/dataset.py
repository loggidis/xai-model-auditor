import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    image_count: int
    label_count: int
    created_at: datetime


class ImageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    filename: str
    has_label: bool


class ImagePage(BaseModel):
    items: list[ImageRead]
    total: int
    limit: int
    offset: int