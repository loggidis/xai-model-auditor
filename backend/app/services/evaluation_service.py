import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.audit import Audit, AuditResult
from app.models.dataset import DatasetImage
from app.models.evaluation import Evaluation
from app.models.user import User
from app.schemas.evaluation import EvaluationCreate, EvaluationRead, EvaluationUpdate


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_notes(notes: str | None) -> str | None:
    return (notes or "").strip() or None


def to_read(db: Session, ev: Evaluation) -> EvaluationRead:
    filename = db.scalar(select(DatasetImage.filename).where(DatasetImage.id == ev.image_id)) or ""
    return EvaluationRead(
        id=ev.id, audit_id=ev.audit_id, image_id=ev.image_id, filename=filename,
        trust_score=ev.trust_score, category=ev.category, notes=ev.notes,
        created_at=ev.created_at, updated_at=ev.updated_at,
    )


def _find(db: Session, audit_id: uuid.UUID, image_id: uuid.UUID, user_id: uuid.UUID) -> Evaluation | None:
    return db.scalar(select(Evaluation).where(
        Evaluation.audit_id == audit_id, Evaluation.image_id == image_id, Evaluation.user_id == user_id))


def _get_owned(db: Session, user: User, audit: Audit, eval_id: uuid.UUID) -> Evaluation:
    ev = db.scalar(select(Evaluation).where(
        Evaluation.id == eval_id, Evaluation.audit_id == audit.id, Evaluation.user_id == user.id))
    if not ev:
        raise HTTPException(404, "Evaluation not found")
    return ev


def upsert_evaluation(db: Session, user: User, audit: Audit, data: EvaluationCreate) -> tuple[Evaluation, bool]:
    """Creates the evaluation, or updates the existing one. Returns (evaluation, created)."""
    result_status = db.scalar(select(AuditResult.status).where(
        AuditResult.audit_id == audit.id, AuditResult.image_id == data.image_id))
    if result_status is None:
        raise HTTPException(404, "Image is not part of this audit")
    if result_status != "done":
        raise HTTPException(409, "Only processed images can be evaluated")

    notes = _clean_notes(data.notes)
    ev = _find(db, audit.id, data.image_id, user.id)
    created = ev is None
    if created:
        ev = Evaluation(audit_id=audit.id, image_id=data.image_id, user_id=user.id,
                        trust_score=data.trust_score, category=data.category, notes=notes)
        db.add(ev)
    else:
        ev.trust_score, ev.category, ev.notes, ev.updated_at = data.trust_score, data.category, notes, _now()
    try:
        db.commit()
    except IntegrityError:                       # two concurrent creates: fall back to an update
        db.rollback()
        ev = _find(db, audit.id, data.image_id, user.id)
        if ev is None:
            raise
        created = False
        ev.trust_score, ev.category, ev.notes, ev.updated_at = data.trust_score, data.category, notes, _now()
        db.commit()
    db.refresh(ev)
    return ev, created


def list_evaluations(db: Session, user: User, audit: Audit, category: str | None,
                     limit: int, offset: int) -> list[EvaluationRead]:
    q = (select(Evaluation, DatasetImage.filename)
         .join(DatasetImage, DatasetImage.id == Evaluation.image_id)
         .where(Evaluation.audit_id == audit.id, Evaluation.user_id == user.id))
    if category:
        q = q.where(Evaluation.category == category)
    rows = db.execute(q.order_by(Evaluation.updated_at.desc()).limit(limit).offset(offset)).all()
    return [
        EvaluationRead(
            id=e.id, audit_id=e.audit_id, image_id=e.image_id, filename=fn, trust_score=e.trust_score,
            category=e.category, notes=e.notes, created_at=e.created_at, updated_at=e.updated_at)
        for e, fn in rows
    ]


def update_evaluation(db: Session, user: User, audit: Audit, eval_id: uuid.UUID,
                      data: EvaluationUpdate) -> Evaluation:
    ev = _get_owned(db, user, audit, eval_id)
    changes = data.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(422, "No fields to update")
    for key in ("trust_score", "category"):
        if key in changes and changes[key] is None:
            raise HTTPException(422, f"'{key}' cannot be null")
    if "trust_score" in changes:
        ev.trust_score = changes["trust_score"]
    if "category" in changes:
        ev.category = changes["category"]
    if "notes" in changes:
        ev.notes = _clean_notes(changes["notes"])
    ev.updated_at = _now()
    db.commit()
    db.refresh(ev)
    return ev


def delete_evaluation(db: Session, user: User, audit: Audit, eval_id: uuid.UUID) -> None:
    ev = _get_owned(db, user, audit, eval_id)
    db.delete(ev)
    db.commit()