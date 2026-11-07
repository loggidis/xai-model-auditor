import logging
import uuid

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.audit import Audit
from app.models.notification import Notification
from app.models.user import User
from app.models.user_settings import UserSettings
from app.models.audit_export import AuditExport
from app.services.progress import progress_counts

log = logging.getLogger(__name__)


def notify_audit(db: Session, audit: Audit, kind: str) -> None:
    """kind: completed | failed | cancelled. Never throws an error (must not interrupt/break the audit)."""
    try:
        prefs = db.get(UserSettings, audit.user_id)
        if prefs is not None and not prefs.notifications_enabled:
            return
        c = progress_counts(db, audit.id, audit.total_images)
        short = str(audit.id)[:8]
        title = {"completed": f"Audit {short} completed",
                 "failed": f"Audit {short} failed",
                 "cancelled": f"Audit {short} cancelled"}[kind]
        message = f"{c['done']}/{c['total']} images processed, {c['failed']} failed."
        if kind == "failed" and audit.error_message:
            message = f"{audit.error_message} ({message})"
        db.add(Notification(user_id=audit.user_id, audit_id=audit.id,
                            type=f"audit_{kind}", title=title, message=message))
        db.commit()
    except Exception:
        log.exception("Notification failed")
        db.rollback()


def list_notifications(db: Session, user: User, unread_only: bool, limit: int, offset: int):
    q = select(Notification).where(Notification.user_id == user.id)
    if unread_only:
        q = q.where(Notification.is_read.is_(False))
    # unread first, then the most recent
    return list(db.scalars(q.order_by(Notification.is_read.asc(), Notification.created_at.desc())
                           .limit(limit).offset(offset)))


def mark_read(db: Session, user: User, notification_id: uuid.UUID) -> Notification:
    n = db.scalar(select(Notification).where(Notification.id == notification_id,
                                             Notification.user_id == user.id))
    if not n:
        raise HTTPException(404, "Notification not found")
    if not n.is_read:
        n.is_read = True
        db.commit()
        db.refresh(n)
    return n


def unread_count(db: Session, user: User) -> int:
    return db.scalar(select(func.count()).select_from(Notification)
                     .where(Notification.user_id == user.id, Notification.is_read.is_(False))) or 0


def notify_export(db: Session, export: AuditExport, kind: str) -> None:
    """kind: ready | failed. Never raises (a notification must not break the export)."""
    try:
        prefs = db.get(UserSettings, export.user_id)
        if prefs is not None and not prefs.notifications_enabled:
            return
        short = str(export.audit_id)[:8]
        if kind == "ready":
            title = f"Export for audit {short} is ready"
            message = f"{export.image_count} flagged images exported."
            if export.skipped_count:
                message += f" {export.skipped_count} skipped (file missing)."
        else:
            title = f"Export for audit {short} failed"
            message = export.error_message or "Unknown error."
        db.add(Notification(user_id=export.user_id, audit_id=export.audit_id,
                            type=f"export_{kind}", title=title, message=message))
        db.commit()
    except Exception:
        log.exception("Export notification failed")
        db.rollback()