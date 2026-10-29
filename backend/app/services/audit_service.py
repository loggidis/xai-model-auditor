import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models.audit import Audit, AuditResult
from app.models.dataset import DatasetImage
from app.models.user import User
from app.schemas.audit import (AuditCreate, AuditRead, HeatmapBrief, HeatmapRead, Progress,
                               ResultDetail, ResultSummary)
from app.services import dataset_service, model_service, settings_service, storage
from app.xai import registry
from app.xai.heatmap_utils import heatmap_rel_path


def create_audit(db: Session, user: User, data: AuditCreate) -> Audit:
    model = model_service.get_owned_model(db, user, data.model_id)        # 404 αν δεν είναι δικό του
    dataset = dataset_service.get_owned_dataset(db, user, data.dataset_id)
    if model.architecture_type is None or model.recommended_layers is None:
        model_service.inspect_model(db, model)                            # μοντέλα του Sprint 2

    methods = list(dict.fromkeys(data.xai_methods))                       # χωρίς διπλότυπα
    resolved = []
    for name in methods:
        try:
            resolved.append(registry.get_method(name))
        except KeyError:
            raise HTTPException(422, f"Unknown XAI method '{name}'")
    for m in resolved:
        if model.architecture_type not in m.applicable_architectures:
            raise HTTPException(422, f"Method '{m.name}' does not support architecture "
                                     f"'{model.architecture_type}'")

    layers = model.recommended_layers
    layer = data.target_layer or next(l["name"] for l in layers if l["default"])
    if layer not in [l["name"] for l in layers]:
        raise HTTPException(422, f"Invalid target_layer '{layer}'. "
                                 f"Valid: {[l['name'] for l in layers]}")

    mask = data.mask or "full_image"
    for m in resolved:
        if mask not in m.supported_masks:
            raise HTTPException(422, f"Mask '{mask}' is not supported by '{m.name}'. "
                                     f"Supported: {m.supported_masks}")

    batch_size = settings_service.get_settings(db, user).batch_size
    image_ids = db.scalars(
        select(DatasetImage.id).where(DatasetImage.dataset_id == dataset.id)
        .order_by(DatasetImage.filename)
    ).all()

    audit = Audit(
        id=uuid.uuid4(), user_id=user.id, model_id=model.id, dataset_id=dataset.id,
        status="pending", total_images=len(image_ids),
        config_json={"xai_methods": methods, "target_layer": layer, "mask": mask,
                     "conf_threshold": data.conf_threshold, "batch_size": batch_size},
    )
    db.add(audit)
    db.flush()
    db.add_all([
        AuditResult(audit_id=audit.id, image_id=img_id, image_index=i, batch_index=i // batch_size)
        for i, img_id in enumerate(image_ids)
    ])
    db.commit()                                   # πρώτα η βάση, μετά το task

    from app.worker.tasks import run_audit
    try:
        audit.celery_task_id = run_audit.delay(str(audit.id)).id
        db.commit()
    except Exception:
        audit.status = "failed"
        audit.error_message = "Task queue unavailable"
        audit.finished_at = datetime.now(timezone.utc)
        db.commit()
        raise HTTPException(503, "Task queue unavailable (is Redis running?)")
    db.refresh(audit)
    return audit


def get_owned_audit(db: Session, user: User, audit_id: uuid.UUID) -> Audit:
    obj = db.scalar(select(Audit).where(Audit.id == audit_id, Audit.user_id == user.id))
    if not obj:
        raise HTTPException(404, "Audit not found")
    return obj


def to_read(db: Session, audits: list[Audit]) -> list[AuditRead]:
    ids = [a.id for a in audits]
    counts: dict[uuid.UUID, dict[str, int]] = {i: {} for i in ids}
    rows = db.execute(
        select(AuditResult.audit_id, AuditResult.status, func.count())
        .where(AuditResult.audit_id.in_(ids))
        .group_by(AuditResult.audit_id, AuditResult.status)
    ).all()
    for aid, st, n in rows:
        counts[aid][st] = n

    out = []
    for a in audits:
        c = counts[a.id]
        done, failed = c.get("done", 0), c.get("failed", 0)
        out.append(AuditRead(
            id=a.id, model_id=a.model_id, dataset_id=a.dataset_id, status=a.status,
            config=a.config_json,
            progress=Progress(
                total=a.total_images, done=done, failed=failed,
                processing=c.get("processing", 0), pending=c.get("pending", 0),
                percent=round(100 * (done + failed) / a.total_images, 1) if a.total_images else 0.0,
            ),
            error_message=a.error_message, created_at=a.created_at,
            started_at=a.started_at, finished_at=a.finished_at,
        ))
    return out


def list_audits(db: Session, user: User, limit: int, offset: int) -> list[AuditRead]:
    audits = list(db.scalars(
        select(Audit).where(Audit.user_id == user.id)
        .order_by(Audit.created_at.desc()).limit(limit).offset(offset)
    ))
    return to_read(db, audits)


def delete_audit(db: Session, user: User, audit_id: uuid.UUID) -> None:
    audit = get_owned_audit(db, user, audit_id)
    if audit.status == "running":
        raise HTTPException(409, "Audit is running (cancel will be added later)")
    db.delete(audit)                              # τα results/heatmaps σβήνονται με CASCADE
    db.commit()
    storage.remove_dir(f"audits/{audit_id}")


def list_results(db: Session, audit: Audit, status: str | None, batch_index: int | None,
                 limit: int, offset: int) -> tuple[list[ResultSummary], int]:
    q = select(AuditResult).where(AuditResult.audit_id == audit.id)
    if status:
        q = q.where(AuditResult.status == status)
    if batch_index is not None:
        q = q.where(AuditResult.batch_index == batch_index)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(
        q.options(selectinload(AuditResult.heatmaps), selectinload(AuditResult.image))
        .order_by(AuditResult.image_index).limit(limit).offset(offset)
    ).all()
    items = [
        ResultSummary(
            image_id=r.image_id, filename=r.image.filename, image_index=r.image_index,
            batch_index=r.batch_index, status=r.status, detection_count=len(r.detections or []),
            heatmaps=[HeatmapBrief(method=h.method, status=h.status) for h in r.heatmaps],
        ) for r in rows
    ]
    return items, total


def _get_result(db: Session, audit: Audit, image_id: uuid.UUID) -> AuditResult:
    r = db.scalar(
        select(AuditResult)
        .where(AuditResult.audit_id == audit.id, AuditResult.image_id == image_id)
        .options(selectinload(AuditResult.heatmaps), selectinload(AuditResult.image))
    )
    if not r:
        raise HTTPException(404, "Result not found")
    return r


def get_result_detail(db: Session, audit: Audit, image_id: uuid.UUID) -> ResultDetail:
    r = _get_result(db, audit, image_id)
    return ResultDetail(
        image_id=r.image_id, filename=r.image.filename, image_index=r.image_index,
        batch_index=r.batch_index, status=r.status, error_message=r.error_message,
        detections=r.detections,
        heatmaps=[
            HeatmapRead(
                method=h.method, status=h.status, runtime_ms=h.runtime_ms,
                error_message=h.error_message, box_focus=h.box_focus,
                url=(f"/api/audits/{audit.id}/results/{r.image_id}/heatmap/{h.method}"
                     if h.status == "done" else None),
            ) for h in r.heatmaps
        ],
    )


def get_heatmap_file(db: Session, audit: Audit, image_id: uuid.UUID, method: str, raw: bool) -> Path:
    r = _get_result(db, audit, image_id)
    h = next((h for h in r.heatmaps if h.method == method and h.status == "done"), None)
    if h is None:
        raise HTTPException(404, "Heatmap not available")
    rel = heatmap_rel_path(audit.id, image_id, method, raw=raw)
    path = storage.abs_path(rel)
    if not path.is_file():
        raise HTTPException(404, "Heatmap file missing")
    return path