import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Integer, BigInteger, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class MLModel(Base):
    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    task: Mapped[str] = mapped_column(String(20))  # detection | classification
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(500))  # σχετικό με STORAGE_DIR
    file_size: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )