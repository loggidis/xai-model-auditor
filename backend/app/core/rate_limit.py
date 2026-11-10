import logging
import time

import redis
from fastapi import HTTPException, Request

from app.core.config import settings

log = logging.getLogger(__name__)
_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
    return _client


def client_ip(request: Request) -> str:
    if settings.TRUST_PROXY_HEADERS:                  # never enable this without a trusted proxy
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def hit(key: str, limit: int, window_s: int) -> int:
    """Counts one request in a fixed window. Returns 0 if allowed, else the seconds to wait.

    Fails open: if Redis is unavailable the request is allowed (and a warning is logged).
    A fixed window can let through up to 2x the limit across a window boundary.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return 0
    now = time.time()
    window = int(now // window_s)
    redis_key = f"rl:{key}:{window}"
    try:
        pipe = _redis().pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, window_s + 1)
        count = pipe.execute()[0]
    except Exception:
        log.warning("Rate limiter unavailable, allowing the request", exc_info=True)
        return 0
    if count <= limit:
        return 0
    return max(1, int((window + 1) * window_s - now))


def enforce(key: str, limit: int, window_s: int) -> None:
    retry_after = hit(key, limit, window_s)
    if retry_after:
        raise HTTPException(429, "Too many requests, please try again later",
                            headers={"Retry-After": str(retry_after)})