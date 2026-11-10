import logging

from app.db.session import SessionLocal
from app.services import cleanup_service
from app.worker.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(name="maintenance.cleanup")
def run_cleanup_task() -> None:
    db = SessionLocal()
    try:
        r = cleanup_service.run_cleanup(db, dry_run=False)
        log.info("Cleanup done: %d refresh tokens, %d verification tokens, %d orphan paths (%d bytes), "
                 "%d skipped as recent, %d errors",
                 r.refresh_tokens_deleted, r.verification_tokens_deleted, len(r.orphan_paths),
                 r.orphan_bytes, r.skipped_recent, len(r.errors))
    finally:
        db.close()