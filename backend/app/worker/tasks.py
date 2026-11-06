import gc
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.audit import Audit, AuditResult, AuditResultHeatmap
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.services import notification_service, storage
from app.services.events import publish
from app.services.progress import progress_counts
from app.worker.celery_app import celery_app

log = logging.getLogger(__name__)

# One audit at a time per worker process: protects GPU/RAM memory even if
# someone starts the worker with more threads.
_DEVICE_LOCK = threading.Lock()


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


def _analyse(res: AuditResult, cfg: dict, maps: dict, focus: dict, any_targets: bool) -> None:
    """Agreement + auto_tag. An error here must NOT fail the image."""
    from app.xai.agreement import compute_agreement
    from app.xai.tagging import TaggingConfig, decide_tag

    tcfg = TaggingConfig(**(cfg.get("tagging") or {}))
    agreement = (compute_agreement(maps, tcfg.iou_threshold, tcfg.iou_weight)
                 if any_targets and len(maps) >= 2 else None)
    score = agreement["score"] if agreement else None
    confs = [d["conf"] for d in (res.detections or [])]
    vals = [v for v in focus.values() if v is not None]
    tag, reason = decide_tag(
        tcfg, has_targets=any_targets, agreement=score,
        mean_conf=(sum(confs) / len(confs)) if confs else None,
        min_box_focus=min(vals) if vals else None,
    )
    res.agreement_score, res.agreement_details = score, agreement
    res.auto_tag, res.auto_tag_reason = tag, reason[:255]


def _process_result(db: Session, audit: Audit, res: AuditResult, img_path, loaded, cfg: dict) -> None:
    """Processes one image. XAI/image errors are recorded in the database and are NOT propagated."""
    from app.xai.heatmap_utils import read_image_rgb, save_heatmap_files
    from app.xai.inference import detect
    from app.xai.registry import get_method

    res.status = "processing"
    res.error_message = None
    res.agreement_score = res.agreement_details = res.auto_tag = res.auto_tag_reason = None
    res.heatmaps.clear()               # cleanup from a previous (interrupted or failed) attempt
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

    method_params = cfg.get("method_params") or {}
    ok, any_targets = 0, False
    maps: dict = {}
    focus: dict = {}
    for name in cfg["xai_methods"]:
        row = AuditResultHeatmap(audit_result_id=res.id, method=name)
        t0 = time.perf_counter()
        try:
            out = get_method(name).run(loaded, image, cfg["target_layer"], cfg["mask"],
                                       conf_threshold=cfg["conf_threshold"],
                                       **method_params.get(name, {}))
            row.file_path = save_heatmap_files(audit.id, res.image_id, name, image, out.heatmap)
            row.box_focus = out.box_focus
            row.status = "done"
            ok += 1
            maps[name], focus[name] = out.heatmap, out.box_focus
            any_targets = any_targets or bool(out.targets)
        except Exception as e:            # only this method fails (e.g. timeout)
            log.exception("Method %s failed on %s", name, res.image_id)
            row.status, row.error_message = "failed", _short(e)
            _free_memory()
        row.runtime_ms = int((time.perf_counter() - t0) * 1000)
        db.add(row)

    res.status = "done" if ok else "failed"
    if not ok:
        res.error_message = "All XAI methods failed"
    else:
        try:
            _analyse(res, cfg, maps, focus, any_targets)
        except Exception:
            log.exception("Agreement/tagging failed on %s", res.image_id)
    res.processed_at = _now()
    db.commit()


def _emit_image_events(db: Session, audit: Audit, res: AuditResult) -> None:
    event = "image_done" if res.status == "done" else "image_failed"
    publish(audit.id, event, {"image_id": str(res.image_id), "image_index": res.image_index,
                              "status": res.status, "error": res.error_message,
                              "auto_tag": res.auto_tag, "agreement_score": res.agreement_score})
    publish(audit.id, "progress",
            {**progress_counts(db, audit.id, audit.total_images), "status": "running"})


def _run(db: Session, audit_id: uuid.UUID) -> None:
    from app.xai.loader import load_model      # lazy, so it can be patched in tests

    audit = db.get(Audit, audit_id)
    if audit is None or audit.status not in ("pending", "running"):
        return                                  # deleted, cancelled or already finished

    started = db.execute(
        update(Audit).where(Audit.id == audit_id, Audit.status.in_(("pending", "running")))
        .values(status="running", started_at=func.coalesce(Audit.started_at, _now()))
    )
    if started.rowcount == 0:
        db.rollback()
        return
    # images left as "processing" by a crash are put back in the queue
    db.execute(update(AuditResult)
               .where(AuditResult.audit_id == audit_id, AuditResult.status == "processing")
               .values(status="pending"))
    db.commit()
    db.refresh(audit)

    model_row = db.get(MLModel, audit.model_id)
    dataset = db.get(Dataset, audit.dataset_id)
    cfg = dict(audit.config_json)
    loaded = load_model(storage.abs_path(model_row.file_path))   # once per audit

    while True:
        batch = db.execute(
            select(AuditResult.id, AuditResult.image_id)
            .where(AuditResult.audit_id == audit_id, AuditResult.status == "pending")
            .order_by(AuditResult.image_index).limit(cfg["batch_size"])
        ).all()
        if not batch:
            break
        for result_id, image_id in batch:
            db.refresh(audit)
            if audit.status == "cancelled":     # cancellation check before each image
                log.info("Audit %s cancelled, stopping", audit_id)
                return
            res = db.get(AuditResult, result_id)
            path = storage.abs_path(f"{dataset.path}/images/{res.image.filename}")
            try:
                _process_result(db, audit, res, path, loaded, cfg)
            except Exception as e:             # database/unexpected error: the image fails, the audit continues
                log.exception("Unexpected error on %s", image_id)
                db.rollback()
                res = db.get(AuditResult, result_id)
                res.status, res.error_message, res.processed_at = "failed", _short(e), _now()
                db.commit()
            _emit_image_events(db, audit, res)

    done = db.scalar(select(func.count()).select_from(AuditResult)
                     .where(AuditResult.audit_id == audit_id, AuditResult.status == "done"))
    if done == 0 and audit.total_images > 0:
        new_status, err = "failed", "All images failed"
    else:
        new_status, err = "completed", None
    # conditional: if it was cancelled in the meantime, nothing changes
    final = db.execute(
        update(Audit).where(Audit.id == audit_id, Audit.status == "running")
        .values(status=new_status, error_message=err, finished_at=_now())
    )
    db.commit()
    if final.rowcount:
        db.refresh(audit)
        publish(audit_id, new_status,
                {**progress_counts(db, audit_id, audit.total_images), "status": new_status, "error": err})
        notification_service.notify_audit(db, audit, new_status)


def _mark_failed(db: Session, audit_id: uuid.UUID, message: str) -> None:
    r = db.execute(
        update(Audit).where(Audit.id == audit_id, Audit.status.in_(("pending", "running")))
        .values(status="failed", error_message=message, finished_at=_now())
    )
    db.commit()
    if r.rowcount:
        audit = db.get(Audit, audit_id)
        publish(audit_id, "failed",
                {**progress_counts(db, audit_id, audit.total_images), "status": "failed", "error": message})
        notification_service.notify_audit(db, audit, "failed")


@celery_app.task(name="audits.run")
def run_audit(audit_id: str) -> None:
    aid = uuid.UUID(audit_id)
    with _DEVICE_LOCK:                          # one audit at a time on this worker
        db = SessionLocal()
        try:
            _run(db, aid)
        except Exception as e:                 # e.g. the model fails to load
            log.exception("Audit %s failed", audit_id)
            db.rollback()
            _mark_failed(db, aid, _short(e))
        finally:
            db.close()
            _free_memory()                     # the model leaves memory (and the GPU)