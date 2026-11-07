import logging
import uuid

from app.db.session import SessionLocal
from app.services import export_service
from app.worker.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(name="exports.run")
def run_export(export_id: str) -> None:
    eid = uuid.UUID(export_id)
    db = SessionLocal()
    try:
        export_service.build_export(db, eid)
    except Exception as e:
        log.exception("Export %s failed", export_id)
        db.rollback()
        export_service.mark_export_failed(db, eid, f"{type(e).__name__}: {e}")
    finally:
        db.close()