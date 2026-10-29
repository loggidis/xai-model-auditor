import gc
import logging
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from app.db.session import SessionLocal
from app.models.audit import Audit, AuditResult, AuditResultHeatmap
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.services import storage
from app.worker.celery_app import celery_app

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _short(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:500]


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _process_result(db: Session, audit: Audit, res: AuditResult, img_path, loaded, cfg: dict) -> None:
    from app.xai.heatmap_utils import read_image_rgb, save_heatmap_files
    from app.xai.inference import detect
    from app.xai.registry import get_method

    res.status = "processing"
    res.error_message = None
    res.heatmaps.clear()              
    db.commit()

    try:
        image = read_image_rgb(img_path)
        res.detections = [d.to_dict() for d in detect(loaded, image, cfg["conf_threshold"])]
    except Exception as e:
        log.exception("Image failed: %s", res.image_id)
        res.status, res.error_message, res.processed_at = "failed", _short(e), _now()
        db.commit()
        _free_memory()
        return

    ok = 0
    for name in cfg["xai_methods"]:
        row = AuditResultHeatmap(audit_result_id=res.id, method=name)
        t0 = time.perf_counter()
        try:
            out = get_method(name).run(loaded, image, cfg["target_layer"], cfg["mask"],
                                       conf_threshold=cfg["conf_threshold"])
            row.file_path = save_heatmap_files(audit.id, res.image_id, name, image, out.heatmap)
            row.box_focus = out.box_focus
            row.status = "done"
            ok += 1
        except Exception as e:
            log.exception("Method %s failed on %s", name, res.image_id)
            row.status, row.error_message = "failed", _short(e)
            _free_memory()
        row.runtime_ms = int((time.perf_counter() - t0) * 1000)
        db.add(row)

    res.status = "done" if ok else "failed"
    if not ok:
        res.error_message = "All XAI methods failed"
    res.processed_at = _now()
    db.commit()


def _run(db: Session, audit_id: uuid.UUID) -> None:
    from app.xai.loader import load_model      # lazy, ώστε να μπορεί να γίνει patch στα tests

    audit = db.get(Audit, audit_id)
    if audit is None or audit.status in ("completed", "failed", "cancelled"):
        return
    model_row = db.get(MLModel, audit.model_id)
    dataset = db.get(Dataset, audit.dataset_id)
    cfg = dict(audit.config_json)

    audit.status = "running"
    audit.started_at = audit.started_at or _now()
    # εικόνες που έμειναν "processing" από crash ξαναμπαίνουν στην ουρά
    db.execute(update(AuditResult)
               .where(AuditResult.audit_id == audit_id, AuditResult.status == "processing")
               .values(status="pending"))
    db.commit()

    loaded = load_model(storage.abs_path(model_row.file_path))   # μία φορά ανά audit

    while True:
        db.refresh(audit)
        if audit.status == "cancelled":       # προετοιμασία για το cancel endpoint
            return
        batch = db.execute(
            select(AuditResult.id, AuditResult.image_id)
            .where(AuditResult.audit_id == audit_id, AuditResult.status == "pending")
            .order_by(AuditResult.image_index).limit(cfg["batch_size"])
        ).all()
        if not batch:
            break
        for result_id, image_id in batch:
            res = db.get(AuditResult, result_id)
            path = storage.abs_path(f"{dataset.path}/images/{res.image.filename}")
            try:
                _process_result(db, audit, res, path, loaded, cfg)
            except Exception as e:             # σφάλμα βάσης/απρόβλεπτο: η εικόνα αποτυγχάνει, το audit συνεχίζει
                log.exception("Unexpected error on %s", image_id)
                db.rollback()
                res = db.get(AuditResult, result_id)
                res.status, res.error_message, res.processed_at = "failed", _short(e), _now()
                db.commit()

    done = db.scalar(select(func.count()).select_from(AuditResult)
                     .where(AuditResult.audit_id == audit_id, AuditResult.status == "done"))
    if done == 0 and audit.total_images > 0:
        audit.status, audit.error_message = "failed", "All images failed"
    else:
        audit.status = "completed"
    audit.finished_at = _now()
    db.commit()


@celery_app.task(name="audits.run")
def run_audit(audit_id: str) -> None:
    aid = uuid.UUID(audit_id)
    db = SessionLocal()
    try:
        _run(db, aid)
    except Exception as e:                     # π.χ. το μοντέλο δεν φορτώνεται
        log.exception("Audit %s failed", audit_id)
        db.rollback()
        audit = db.get(Audit, aid)
        if audit:
            audit.status, audit.error_message, audit.finished_at = "failed", _short(e), _now()
            db.commit()
    finally:
        db.close()