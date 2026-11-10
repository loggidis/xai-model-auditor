import uuid

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import email_service

client = TestClient(app)


def test_verification_email_is_multipart_with_working_link(monkeypatch, outbox):
    monkeypatch.setattr(settings, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(settings, "PUBLIC_API_URL", "http://api.test:8000")
    monkeypatch.setattr(settings, "VERIFICATION_URL_TEMPLATE", None)
    email = f"mail_{uuid.uuid4().hex[:8]}@example.com"
    assert client.post("/api/auth/register", json={"email": email, "password": "password123"}).status_code == 201

    assert len(outbox) == 1
    msg = outbox[0]
    assert msg["To"] == email and msg["Subject"]
    text = msg.get_body(preferencelist=("plain",)).get_content()
    html = msg.get_body(preferencelist=("html",)).get_content()
    prefix = "http://api.test:8000/api/auth/verify-email?token="
    link = next(word for word in text.split() if word.startswith(prefix))
    assert link in html
    token = link[len(prefix):]

    page = client.get(f"/api/auth/verify-email?token={token}")           # opening the link does NOT use the token
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert "Confirm email" in page.text
    assert client.post("/api/auth/verify-email", json={"token": token}).status_code == 200
    assert client.post("/api/auth/verify-email", json={"token": token}).status_code == 400   # single use


def test_custom_verification_url_template(monkeypatch, outbox):
    monkeypatch.setattr(settings, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(settings, "VERIFICATION_URL_TEMPLATE", "https://app.test/verify?t={token}")
    assert email_service.send_verification_email("someone@example.com", "TOKEN123") is True
    body = outbox[0].get_body(preferencelist=("plain",)).get_content()
    assert "https://app.test/verify?t=TOKEN123" in body


def test_not_configured_sends_nothing(monkeypatch, outbox):
    monkeypatch.setattr(settings, "SMTP_HOST", None)
    monkeypatch.setattr(settings, "EMAIL_DEV_MODE", False)
    assert email_service.send_email("someone@example.com", "s", "t") is False
    assert outbox == []
    with pytest.raises(email_service.EmailError):
        email_service.send_email("someone@example.com", "s", "t", raise_errors=True)