import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    description: str | None
    task: str
    architecture_type: str | None
    num_classes: int | None
    input_size: int | None
    class_names: list[str] | None
    original_filename: str
    file_size: int
    created_at: datetime