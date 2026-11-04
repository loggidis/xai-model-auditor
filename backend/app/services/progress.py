from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models.audit import AuditResult


def progress_counts(db: Session, audit_id, total: int) -> dict:
    c = dict(db.execute(
        select(AuditResult.status, func.count())
        .where(AuditResult.audit_id == audit_id).group_by(AuditResult.status)
    ).all())
    done, failed = c.get("done", 0), c.get("failed", 0)
    return {
        "total": total, "done": done, "failed": failed,
        "processing": c.get("processing", 0), "pending": c.get("pending", 0),
        "percent": round(100 * (done + failed) / total, 1) if total else 0.0,
    }