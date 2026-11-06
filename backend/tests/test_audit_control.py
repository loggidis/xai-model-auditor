import io
import uuid
import zipfile
from dataclasses import asdict

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

pytest.importorskip("ultralytics")
from ultralytics import YOLO

from app.db.session import SessionLocal
from app.main import app
from app.models.audit import AuditResult
from app.models.ml_model import MLModel
from app.services import audit_service
from app.worker import tasks
from app.xai.loader import from_yolo

client = TestClient(app)


def _auth():
    email = "control_user@example.com"
    client.post("/api/auth/register", json={"email": email, "password": "password123"})
    r = client.post("/api/auth/login", data={"username": email, "password": "password123"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _jpg(seed: int) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture
def setup(monkeypatch):
    h = _auth()
    me = client.get("/api/auth/me", headers=h).json()
    tiny = from_yolo(YOLO("yolov8n.yaml"))
    monkeypatch.setattr("app.xai.loader.load_model", lambda path: tiny)
    monkeypatch.setattr(tasks.run_audit, "delay", lambda *a, **k: type("R", (), {"id": "test"})())
    monkeypatch.setattr(audit_service.celery_app.control, "revoke", lambda *a, **k: None)

    with SessionLocal() as db:
        m = MLModel(
            user_id=uuid.UUID(me["id"]), name="t", task="detection", original_filename="x.pt",
            file_path="models/none/model.pt", file_size=1, architecture_type="yolo_detect",
            num_classes=tiny.info.num_classes, input_size=tiny.info.input_size,
            class_names=tiny.info.class_names,
            recommended_layers=[asdict(l) for l in tiny.info.recommended_layers],
        )
        db.add(m)
        db.commit()
        model_id = str(m.id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(3):
            z.writestr(f"img{i}.jpg", _jpg(i))
    buf.seek(0)
    ds = client.post("/api/datasets", headers=h, data={"name": "t"},
                     files={"file": ("d.zip", buf, "application/zip")}).json()
    r = client.post("/api/audits", headers=h, json={
        "model_id": model_id, "dataset_id": ds["id"], "xai_methods": ["gradcam_pp"],
        "conf_threshold": 0.01})
    assert r.status_code == 201
    yield h, r.json()["id"], ds["id"], model_id

    # cleanup
    client.delete(f"/api/audits/{r.json()['id']}", headers=h)
    client.delete(f"/api/datasets/{ds['id']}", headers=h)
    client.delete(f"/api/models/{model_id}", headers=h)


def test_cancel_then_retry_resumes(setup):
    h, audit_id, _, _ = setup
    before = client.get("/api/notifications/unread-count", headers=h).json()["count"]

    r = client.post(f"/api/audits/{audit_id}/cancel", headers=h)
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert client.post(f"/api/audits/{audit_id}/cancel", headers=h).status_code == 409

    with SessionLocal() as db:                       # The worker sees "cancelled" and does nothing
        tasks._run(db, uuid.UUID(audit_id))
    a = client.get(f"/api/audits/{audit_id}", headers=h).json()
    assert a["status"] == "cancelled" and a["progress"]["pending"] == 3

    notes = client.get("/api/notifications", headers=h).json()
    assert notes[0]["type"] == "audit_cancelled" and notes[0]["is_read"] is False
    assert client.get("/api/notifications/unread-count", headers=h).json()["count"] == before + 1
    r = client.patch(f"/api/notifications/{notes[0]['id']}/read", headers=h)
    assert r.status_code == 200 and r.json()["is_read"] is True
    assert client.get("/api/notifications/unread-count", headers=h).json()["count"] == before

    r = client.post(f"/api/audits/{audit_id}/retry", headers=h)     # The worker resumes the pending tasks
    assert r.status_code == 202 and r.json()["status"] == "pending"
    with SessionLocal() as db:
        tasks._run(db, uuid.UUID(audit_id))
    a = client.get(f"/api/audits/{audit_id}", headers=h).json()
    assert a["status"] == "completed" and a["progress"]["done"] == 3
    assert client.get("/api/notifications", headers=h).json()[0]["type"] == "audit_completed"


def test_retry_only_failed(setup):
    h, audit_id, _, _ = setup
    with SessionLocal() as db:
        tasks._run(db, uuid.UUID(audit_id))
    assert client.post(f"/api/audits/{audit_id}/retry", headers=h).status_code == 409   # No failed tasks

    with SessionLocal() as db:                       # Simulate a failed image
        db.execute(update(AuditResult)
                   .where(AuditResult.audit_id == uuid.UUID(audit_id), AuditResult.image_index == 1)
                   .values(status="failed", error_message="boom"))
        db.commit()
    r = client.post(f"/api/audits/{audit_id}/retry", headers=h)
    assert r.status_code == 202 and r.json()["progress"]["pending"] == 1
    with SessionLocal() as db:
        tasks._run(db, uuid.UUID(audit_id))
    a = client.get(f"/api/audits/{audit_id}", headers=h).json()
    assert a["status"] == "completed" and a["progress"]["done"] == 3 and a["progress"]["failed"] == 0


def test_delete_blocked_while_running(setup):
    h, audit_id, _, _ = setup
    with SessionLocal() as db:
        from app.models.audit import Audit
        db.execute(update(Audit).where(Audit.id == uuid.UUID(audit_id)).values(status="running"))
        db.commit()
    assert client.delete(f"/api/audits/{audit_id}", headers=h).status_code == 409
    assert client.post(f"/api/audits/{audit_id}/cancel", headers=h).status_code == 200