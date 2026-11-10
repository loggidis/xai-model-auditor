import logging
import uuid

from app.db.session import SessionLocal
from app.services import summary_service
from app.worker.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(name="summaries.run")
def run_summary(audit_id: str) -> None:
    aid = uuid.UUID(audit_id)
    db = SessionLocal()
    try:
        summary_service.generate_summary(db, aid)
    except Exception as e:
        log.exception("Summary task for audit %s failed", audit_id)
        db.rollback()
        summary_service.mark_failed(db, aid, f"{type(e).__name__}: {e}")
    finally:
        db.close()