import json
import logging

import redis

from app.core.config import settings

log = logging.getLogger(__name__)
_client: redis.Redis | None = None


def channel(audit_id) -> str:
    return f"audit:{audit_id}"


def _get() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=2)
    return _client


def publish(audit_id, event: str, data: dict) -> None:
    """Never throws an error: if Redis is down, only the live event is lost, not the job."""
    try:
        _get().publish(channel(audit_id), json.dumps({"event": event, "data": data}, default=str))
    except Exception as e:
        log.warning("Event publish failed (%s): %s", event, e)