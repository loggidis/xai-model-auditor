from fastapi import Depends, Request

from app.api.deps import get_current_user
from app.core.rate_limit import client_ip, enforce
from app.models.user import User


def limit_by_ip(name: str, limit: int, window_s: int):
    """Dependency factory: at most `limit` requests per `window_s` seconds per client IP."""
    def dependency(request: Request) -> None:
        enforce(f"{name}:ip:{client_ip(request)}", limit, window_s)
    return dependency


def limit_by_user(name: str, limit: int, window_s: int):
    """Dependency factory: at most `limit` requests per `window_s` seconds per authenticated user."""
    def dependency(user: User = Depends(get_current_user)) -> None:
        enforce(f"{name}:user:{user.id}", limit, window_s)
    return dependency