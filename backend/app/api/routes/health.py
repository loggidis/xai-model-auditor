import redis
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.audit import Audit
from app.models.user import User
from app.worker.celery_app import celery_app

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health():
    return {"status": "ok"}


@router.get("/queue")
def queue_health(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    redis_state, pending, workers = "down", None, []
    try:
        r = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        r.ping()
        redis_state, pending = "up", r.llen("celery")
        workers = sorted((celery_app.control.inspect(timeout=1).ping() or {}).keys())
    except Exception:
        pass
    counts = dict(db.execute(
        select(Audit.status, func.count()).where(Audit.status.in_(("pending", "running")))
        .group_by(Audit.status)
    ).all())
    return {
        "redis": redis_state,
        "workers": workers,
        "pending_tasks": pending,                 # tasks που περιμένουν στο Redis
        "audits": {"pending": counts.get("pending", 0), "running": counts.get("running", 0)},
    }