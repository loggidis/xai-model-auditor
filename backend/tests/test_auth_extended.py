import random
import uuid

import pytest
import redis
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import email_service

client = TestClient(app)


def _email() -> str:
    return f"auth_{uuid.uuid4().hex[:10]}@example.com"


def _register(email: str):
    return client.post("/api/auth/register", json={"email": email, "password": "password123"})


def _login(email: str, password: str = "password123", headers: dict | None = None):
    return client.post("/api/auth/login", data={"username": email, "password": password}, headers=headers)


def _refresh(token: str):
    return client.post("/api/auth/refresh", json={"refresh_token": token})


@pytest.fixture
def mailbox(monkeypatch):
    """Captures the verification tokens instead of sending emails."""
    sent: dict[str, str] = {}
    monkeypatch.setattr(email_service, "send_verification_email", lambda to, token: sent.__setitem__(to, token))
    return sent


def test_refresh_rotation_and_reuse_detection():
    email = _email()
    assert _register(email).status_code == 201
    t1 = _login(email).json()
    assert t1["refresh_token"] and t1["expires_in"] > 0

    r = _refresh(t1["refresh_token"])
    assert r.status_code == 200
    t2 = r.json()
    assert t2["refresh_token"] != t1["refresh_token"]
    h = {"Authorization": f"Bearer {t2['access_token']}"}
    assert client.get("/api/auth/me", headers=h).status_code == 200

    reuse = _refresh(t1["refresh_token"])               # an already rotated token is presented again
    assert reuse.status_code == 401 and reuse.json()["error"]["code"] == "unauthorized"
    assert _refresh(t2["refresh_token"]).status_code == 401     # the whole family was revoked


def test_logout_and_logout_all():
    email = _email()
    _register(email)
    a, b = _login(email).json(), _login(email).json()
    assert client.post("/api/auth/logout", json={"refresh_token": a["refresh_token"]}).status_code == 204
    assert client.post("/api/auth/logout", json={"refresh_token": a["refresh_token"]}).status_code == 204   # idempotent
    assert _refresh(a["refresh_token"]).status_code == 401
    assert _refresh(b["refresh_token"]).status_code == 200      # other logins are not affected

    c, d = _login(email).json(), _login(email).json()
    h = {"Authorization": f"Bearer {c['access_token']}"}
    assert client.post("/api/auth/logout-all", headers=h).status_code == 204
    assert _refresh(c["refresh_token"]).status_code == 401
    assert _refresh(d["refresh_token"]).status_code == 401


def test_email_verification_flow(monkeypatch, mailbox):
    monkeypatch.setattr(settings, "REQUIRE_EMAIL_VERIFICATION", True)
    email = _email()
    r = _register(email)
    assert r.status_code == 201 and r.json()["email_verified"] is False
    token = mailbox[email]

    assert _login(email).status_code == 403                     # not verified yet
    bad = client.post("/api/auth/verify-email", json={"token": "x" * 30})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "bad_request"

    assert client.post("/api/auth/verify-email", json={"token": token}).status_code == 200
    login = _login(email)
    assert login.status_code == 200
    h = {"Authorization": f"Bearer {login.json()['access_token']}"}
    assert client.get("/api/auth/me", headers=h).json()["email_verified"] is True
    assert client.post("/api/auth/verify-email", json={"token": token}).status_code == 400   # single use


def test_resend_verification(mailbox):
    email = _email()
    _register(email)
    first = mailbox[email]
    r = client.post("/api/auth/resend-verification", json={"email": email})
    assert r.status_code == 202
    second = mailbox[email]
    assert second != first
    assert client.post("/api/auth/verify-email", json={"token": first}).status_code == 400    # old link invalidated
    assert client.post("/api/auth/verify-email", json={"token": second}).status_code == 200
    # unknown email: same answer, nothing sent
    r = client.post("/api/auth/resend-verification", json={"email": _email()})
    assert r.status_code == 202


def test_unified_error_format():
    r = client.get("/api/audits")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized" and r.headers["www-authenticate"] == "Bearer"

    r = client.post("/api/auth/register", json={"email": "not-an-email", "password": "password123"})
    body = r.json()
    assert r.status_code == 422 and body["error"]["code"] == "validation_error"
    assert body["error"]["details"][0]["field"] == "email"

    r = client.get("/api/does-not-exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_login_rate_limit(monkeypatch):
    try:
        redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1).ping()
    except Exception:
        pytest.skip("Redis is not available")
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)      # a random IP per test run
    headers = {"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4))}
    email = _email()
    limit = settings.RATE_LIMIT_LOGIN_PER_MINUTE

    responses = [_login(email, "wrong-password", headers) for _ in range(limit * 2 + 1)]
    codes = [r.status_code for r in responses]
    assert 429 in codes                                              # (twice the limit covers a window boundary)
    blocked = responses[codes.index(429)]
    assert blocked.headers["retry-after"].isdigit() and int(blocked.headers["retry-after"]) >= 1
    assert blocked.json()["error"]["code"] == "rate_limited"