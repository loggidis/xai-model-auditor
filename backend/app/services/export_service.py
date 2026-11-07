import csv
import io
import os
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.models.audit import Audit
from app.models.audit_export import AuditExport
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.user import User
from app.schemas.report import ExportRead, ExportRequest
from app.services import notification_service, report_service, storage


def _now() -> datetime:
    return datetime.now(timezone.utc)


def to_read(e: AuditExport) -> ExportRead:
    return ExportRead(
        id=e.id, audit_id=e.audit_id, status=e.status, params=e.params, image_count=e.image_count,
        skipped_count=e.skipped_count, file_size=e.file_size, error_message=e.error_message,
        created_at=e.created_at, finished_at=e.finished_at,
        download_url=(f"/api/audits/{e.audit_id}/report/export/{e.id}/download"
                      if e.status == "ready" else None),
    )


# ---------- API side ----------

def create_export(db: Session, user: User, audit: Audit, params: ExportRequest) -> AuditExport:
    busy = db.scalar(select(func.count()).select_from(AuditExport).where(
        AuditExport.audit_id == audit.id, AuditExport.status.in_(("pending", "running"))))
    if busy:
        raise HTTPException(409, "An export is already in progress for this audit")
    if not report_service.flagged_items(db, audit, params.trust_below, params.include_unevaluated):
        raise HTTPException(409, "No flagged images match these criteria")

    export = AuditExport(audit_id=audit.id, user_id=user.id, status="pending", params=params.model_dump())
    db.add(export)
    db.commit()                                   # database first, then the task

    from app.worker.export_tasks import run_export
    try:
        export.celery_task_id = run_export.delay(str(export.id)).id
        db.commit()
    except Exception:
        export.status, export.error_message, export.finished_at = "failed", "Task queue unavailable", _now()
        db.commit()
        raise HTTPException(503, "Task queue unavailable (is Redis running?)")
    db.refresh(export)
    return export


def list_exports(db: Session, audit: Audit) -> list[AuditExport]:
    return list(db.scalars(
        select(AuditExport).where(AuditExport.audit_id == audit.id).order_by(AuditExport.created_at.desc())))


def get_export(db: Session, audit: Audit, export_id: uuid.UUID) -> AuditExport:
    e = db.scalar(select(AuditExport).where(AuditExport.id == export_id, AuditExport.audit_id == audit.id))
    if not e:
        raise HTTPException(404, "Export not found")
    return e


def export_file(e: AuditExport) -> Path:
    if e.status != "ready" or not e.file_path:
        raise HTTPException(409, f"Export is '{e.status}', not ready")
    path = storage.abs_path(e.file_path)
    if not path.is_file():
        raise HTTPException(404, "Export file missing")
    return path


def delete_export(db: Session, audit: Audit, export_id: uuid.UUID) -> None:
    e = get_export(db, audit, export_id)
    if e.status == "running":
        raise HTTPException(409, "Export is running, try again when it finishes")
    rel = e.file_path
    db.delete(e)
    db.commit()
    if rel:
        storage.abs_path(rel).unlink(missing_ok=True)


# ---------- worker side ----------

def _predictions_txt(src: Path, detections: list[dict]) -> str:
    """Model predictions in YOLO format: 'class cx cy w h conf' (normalised, conf as 6th column)."""
    from app.xai.heatmap_utils import read_image_rgb     # lazy: opencv
    h, w = read_image_rgb(src).shape[:2]
    clamp = lambda v: min(max(v, 0.0), 1.0)
    lines = []
    for d in detections or []:
        x1, y1, x2, y2 = d["box"]
        lines.append(" ".join([
            str(int(d["class_id"])),
            f"{clamp((x1 + x2) / 2 / w):.6f}", f"{clamp((y1 + y2) / 2 / h):.6f}",
            f"{clamp((x2 - x1) / w):.6f}", f"{clamp((y2 - y1) / h):.6f}",
            f"{float(d['conf']):.4f}",
        ]))
    return "\n".join(lines) + ("\n" if lines else "")


def _write_zip(tmp: Path, items: list, dataset_path: str, class_names: list[str],
               include_predictions: bool) -> tuple[int, int]:
    included = skipped = 0
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(["image", "reason", "category", "trust_score", "auto_tag", "agreement_score",
                     "detection_count", "mean_confidence", "has_label", "notes"])
    with zipfile.ZipFile(tmp, "w") as zf:
        for it in items:
            src = storage.abs_path(f"{dataset_path}/images/{it.filename}")
            if not src.is_file():
                skipped += 1
                continue
            stem = Path(it.filename).stem
            zf.write(src, f"images/{it.filename}", compress_type=zipfile.ZIP_STORED)   # already compressed
            label = storage.abs_path(f"{dataset_path}/labels/{stem}.txt")
            has_label = it.has_label and label.is_file()
            if has_label:
                zf.write(label, f"labels/{stem}.txt", compress_type=zipfile.ZIP_DEFLATED)
            if include_predictions:
                zf.writestr(f"predictions/{stem}.txt", _predictions_txt(src, it.detections),
                            compress_type=zipfile.ZIP_DEFLATED)
            writer.writerow([it.filename, it.reason, it.category or "", it.trust_score if it.trust_score is not None else "",
                             it.auto_tag or "", it.agreement_score if it.agreement_score is not None else "",
                             it.detection_count, it.mean_confidence if it.mean_confidence is not None else "",
                             int(has_label), it.notes or ""])
            included += 1
        zf.writestr("classes.txt", "\n".join(class_names) + "\n", compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("metadata.csv", buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return included, skipped


def build_export(db: Session, export_id: uuid.UUID) -> None:
    """Runs inside the Celery worker. Raises on failure (the task marks the export as failed)."""
    started = db.execute(
        update(AuditExport).where(AuditExport.id == export_id, AuditExport.status.in_(("pending", "running")))
        .values(status="running"))
    db.commit()
    if started.rowcount == 0:
        return                                      # deleted, or already finished

    export = db.get(AuditExport, export_id)
    audit = db.get(Audit, export.audit_id)
    dataset = db.get(Dataset, audit.dataset_id)
    model = db.get(MLModel, audit.model_id)
    params = export.params

    items = report_service.flagged_items(db, audit, params["trust_below"], params["include_unevaluated"])
    if not items:
        raise ValueError("No flagged images match these criteria")

    rel = f"exports/{audit.id}/{export.id}.zip"
    path = storage.abs_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".zip.part")
    try:
        included, skipped = _write_zip(tmp, items, dataset.path, model.class_names or [],
                                       bool(params.get("include_predictions")))
        if included == 0:
            raise ValueError("None of the flagged image files could be found")
        size = tmp.stat().st_size
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if not db.scalar(select(func.count()).select_from(AuditExport).where(AuditExport.id == export_id)):
        path.unlink(missing_ok=True)                # the export was deleted while it was running
        return
    db.execute(update(AuditExport).where(AuditExport.id == export_id).values(
        status="ready", file_path=rel, file_size=size, image_count=included, skipped_count=skipped,
        error_message=None, finished_at=_now()))
    db.commit()
    db.refresh(export)
    notification_service.notify_export(db, export, "ready")


def mark_export_failed(db: Session, export_id: uuid.UUID, message: str) -> None:
    r = db.execute(
        update(AuditExport).where(AuditExport.id == export_id, AuditExport.status.in_(("pending", "running")))
        .values(status="failed", error_message=message[:500], finished_at=_now()))
    db.commit()
    if r.rowcount:
        export = db.get(AuditExport, export_id)
        notification_service.notify_export(db, export, "failed")