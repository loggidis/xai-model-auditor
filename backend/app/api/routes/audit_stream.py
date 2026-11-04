import json
import uuid

import jwt
import redis
import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import decode_token
from app.db.session import SessionLocal
from app.models.audit import Audit
from app.models.user import User
from app.services.events import channel
from app.services.progress import progress_counts

router = APIRouter(prefix="/audits", tags=["audits"])
TERMINAL = {"completed", "failed", "cancelled"}


def _user_id(request: Request, token: str | None) -> uuid.UUID:
    """Token από header ή από ?token= (το EventSource δεν στέλνει headers)."""
    raw = token
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        raw = auth[7:]
    try:
        payload = decode_token(raw or "")
        if payload.get("type") != "access":
            raise ValueError
        return uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(401, "Invalid or expired token")


def _load_snapshot(audit_id: uuid.UUID, user_id: uuid.UUID) -> tuple[str, dict]:
    with SessionLocal() as db:                      # σύντομη session, δεν μένει ανοιχτή στο stream
        user = db.get(User, user_id)
        if not user or not user.is_active:
            raise HTTPException(401, "Invalid or expired token")
        audit = db.scalar(select(Audit).where(Audit.id == audit_id, Audit.user_id == user_id))
        if not audit:
            raise HTTPException(404, "Audit not found")
        return audit.status, progress_counts(db, audit.id, audit.total_images)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _close(pubsub, client) -> None:
    for coro in (pubsub.aclose(), client.aclose()):
        try:
            await coro
        except Exception:
            pass


@router.get("/{audit_id}/stream")
async def stream_audit(audit_id: uuid.UUID, request: Request, token: str | None = Query(None)):
    user_id = _user_id(request, token)

    client = aioredis.Redis.from_url(settings.REDIS_URL)
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(channel(audit_id))                         # 1. πρώτα subscribe
        status, counts = await run_in_threadpool(_load_snapshot, audit_id, user_id)   # 2. μετά snapshot
    except redis.exceptions.RedisError:
        await _close(pubsub, client)
        raise HTTPException(503, "Event stream unavailable (is Redis running?)")
    except BaseException:
        await _close(pubsub, client)
        raise

    async def events():
        try:
            yield _sse("progress", {**counts, "status": status})
            if status in TERMINAL:
                yield _sse(status, {**counts, "status": status})
                return
            while True:
                if await request.is_disconnected():
                    break
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15)
                if msg is None:
                    yield ": keep-alive\n\n"             # κρατά τη σύνδεση ζωντανή
                    continue
                payload = json.loads(msg["data"])
                yield _sse(payload["event"], payload["data"])
                if payload["event"] in TERMINAL:
                    break
        finally:
            await _close(pubsub, client)

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )