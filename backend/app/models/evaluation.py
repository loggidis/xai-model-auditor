import uuid
from datetime import datetime, timezone

from sqlalchemy import (CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Evaluation(Base):
    """A user's trust rating and category for one image of one audit (one row per audit/image/user)."""
    __tablename__ = "evaluations"
    __table_args__ = (
        UniqueConstraint("audit_id", "image_id", "user_id", name="uq_evaluation_audit_image_user"),
        CheckConstraint("trust_score >= 0 AND trust_score <= 100", name="ck_evaluation_trust_range"),
        CheckConstraint(
            "category IN ('correct_right_reason', 'correct_wrong_reason', 'wrong_prediction')",
            name="ck_evaluation_category",
        ),
        Index("ix_evaluations_audit_category", "audit_id", "category"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audits.id", ondelete="CASCADE")
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_images.id", ondelete="CASCADE")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    trust_score: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(30))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)