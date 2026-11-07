import io
import uuid
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.session import SessionLocal
from app.main import app
from app.models.audit import AuditResult
from app.models.ml_model import MLModel
from app.services import export_service
from app.worker import export_tasks, tasks

client = TestClient(app)

LAYERS = [
    {"name": "model.15", "description": "P3", "stride": 8, "default": False},
    {"name": "model.18", "description": "P4", "stride": 16, "default": True},
    {"name": "model.21", "description": "P5", "stride": 32, "default": False},
]


def _det(name: str, class_id: int, conf: float) -> dict:
    return {"box": [10.0, 10.0, 50.0, 60.0], "class_id": class_id, "class_name": name, "conf": conf}


# (auto_tag, agreement_score, detections) per image_index (images are sorted by filename)
SPEC = [
    ("consistent", 0.9, [_det("apple", 0, 0.9)]),
    ("needs_review", 0.1, [_det("pear", 1, 0.5)]),
    ("consistent", 0.8, [_det("apple", 0, 0.7)]),
]


def _auth():
    email = "reports_user@example.com"
    client.post("/api/auth/register", json={"email": email, "password": "password123"})
    r = client.post("/api/auth/login", data={"username": email, "password": "password123"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _jpg(seed: int) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture
def audit_env(monkeypatch):
    h = _auth()
    me = client.get("/api/auth/me", headers=h).json()
    fake_task = lambda *a, **k: type("R", (), {"id": "test"})()
    monkeypatch.setattr(tasks.run_audit, "delay", fake_task)
    monkeypatch.setattr(export_tasks.run_export, "delay", fake_task)

    with SessionLocal() as db:
        m = MLModel(
            user_id=uuid.UUID(me["id"]), name="t", task="detection", original_filename="x.pt",
            file_path="models/none/model.pt", file_size=1, architecture_type="yolo_detect",
            num_classes=2, input_size=640, class_names=["apple", "pear"], recommended_layers=LAYERS,
        )
        db.add(m)
        db.commit()
        model_id = str(m.id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(3):
            z.writestr(f"img{i}.jpg", _jpg(i))
        z.writestr("labels/img0.txt", "0 0.5 0.5 0.2 0.2")
        z.writestr("labels/img1.txt", "1 0.5 0.5 0.2 0.2")
    buf.seek(0)
    ds = client.post("/api/datasets", headers=h, data={"name": "t"},
                     files={"file": ("d.zip", buf, "application/zip")}).json()

    r = client.post("/api/audits", headers=h, json={
        "model_id": model_id, "dataset_id": ds["id"], "xai_methods": ["gradcam_pp"]})
    assert r.status_code == 201
    audit_id = r.json()["id"]

    with SessionLocal() as db:                       # simulate a finished audit
        rows = db.scalars(select(AuditResult).where(AuditResult.audit_id == uuid.UUID(audit_id))
                          .order_by(AuditResult.image_index)).all()
        for row, (tag, agreement, dets) in zip(rows, SPEC):
            row.status, row.auto_tag, row.agreement_score, row.detections = "done", tag, agreement, dets
        db.commit()

    items = client.get(f"/api/datasets/{ds['id']}/images", headers=h).json()["items"]
    yield h, audit_id, {i["filename"]: i["id"] for i in items}

    for e in client.get(f"/api/audits/{audit_id}/report/export", headers=h).json():
        client.delete(f"/api/audits/{audit_id}/report/export/{e['id']}", headers=h)
    client.delete(f"/api/audits/{audit_id}", headers=h)
    client.delete(f"/api/datasets/{ds['id']}", headers=h)
    client.delete(f"/api/models/{model_id}", headers=h)


def _by_key(slices: list[dict]) -> dict:
    return {s["key"]: s["count"] for s in slices}


def test_evaluations_report_flagged_export(audit_env):
    h, audit_id, ids = audit_env
    base = f"/api/audits/{audit_id}"

    # --- evaluations: upsert, validation, patch ---
    r = client.post(f"{base}/evaluations", headers=h, json={
        "image_id": ids["img0.jpg"], "trust_score": 80, "category": "correct_right_reason"})
    assert r.status_code == 201
    first_id = r.json()["id"]
    r = client.post(f"{base}/evaluations", headers=h, json={
        "image_id": ids["img0.jpg"], "trust_score": 90, "category": "correct_right_reason"})
    assert r.status_code == 200 and r.json()["id"] == first_id and r.json()["trust_score"] == 90
    r = client.post(f"{base}/evaluations", headers=h, json={
        "image_id": ids["img2.jpg"], "trust_score": 20, "category": "wrong_prediction", "notes": "bad"})
    assert r.status_code == 201

    bad = {"image_id": ids["img1.jpg"], "trust_score": 50, "category": "correct_right_reason"}
    assert client.post(f"{base}/evaluations", headers=h, json={**bad, "trust_score": 101}).status_code == 422
    assert client.post(f"{base}/evaluations", headers=h, json={**bad, "category": "nope"}).status_code == 422
    assert client.post(f"{base}/evaluations", headers=h,
                       json={**bad, "image_id": str(uuid.uuid4())}).status_code == 404

    evals = client.get(f"{base}/evaluations", headers=h).json()
    assert len(evals) == 2
    ev2 = next(e for e in evals if e["filename"] == "img2.jpg")
    r = client.patch(f"{base}/evaluations/{ev2['id']}", headers=h, json={"notes": "wrong object"})
    assert r.status_code == 200 and r.json()["notes"] == "wrong object" and r.json()["trust_score"] == 20
    assert client.patch(f"{base}/evaluations/{ev2['id']}", headers=h, json={}).status_code == 422
    assert client.patch(f"{base}/evaluations/{ev2['id']}", headers=h,
                        json={"trust_score": 101}).status_code == 422

    # --- report ---
    rep = client.get(f"{base}/report", headers=h).json()
    assert rep["processed_images"] == 3 and rep["evaluated_images"] == 2
    tags = _by_key(rep["by_auto_tag"])
    assert tags["consistent"] == 2 and tags["needs_review"] == 1
    assert _by_key(rep["by_category"]) == {"correct_right_reason": 1, "correct_wrong_reason": 0,
                                           "wrong_prediction": 1, "unevaluated": 1}
    assert rep["avg_confidence"] == pytest.approx(0.7)
    assert rep["avg_trust"] == pytest.approx(55.0)
    assert rep["avg_agreement"] == pytest.approx(0.6)
    classes = {c["class_name"]: c for c in rep["classes"]}
    assert classes["apple"]["detections"] == 2 and classes["apple"]["avg_confidence"] == pytest.approx(0.8)
    assert classes["apple"]["avg_trust"] == pytest.approx(55.0)
    assert classes["pear"]["detections"] == 1 and classes["pear"]["avg_trust"] is None

    # --- flagged: ascending trust, agreement * 100 as fallback ---
    fl = client.get(f"{base}/report/flagged", headers=h).json()
    assert fl["total"] == 2
    assert [i["filename"] for i in fl["items"]] == ["img1.jpg", "img2.jpg"]
    assert [i["score_source"] for i in fl["items"]] == ["agreement", "evaluation"]
    assert [i["reason"] for i in fl["items"]] == ["auto:needs_review", "user:wrong_prediction"]
    assert fl["items"][0]["sort_score"] == pytest.approx(10.0)
    fl = client.get(f"{base}/report/flagged?trust_below=95", headers=h).json()
    assert [i["filename"] for i in fl["items"]] == ["img1.jpg", "img2.jpg", "img0.jpg"]
    assert fl["items"][2]["reason"] == "user:low_trust"
    assert client.get(f"{base}/report/flagged?include_unevaluated=false", headers=h).json()["total"] == 1

    # --- global report ---
    g = client.get("/api/reports", headers=h).json()
    assert g["audits_total"] >= 1 and any(a["audit_id"] == audit_id for a in g["audits"])

    # --- export (job) ---
    r = client.post(f"{base}/report/export", headers=h, json={"include_predictions": True})
    assert r.status_code == 202 and r.json()["status"] == "pending"
    export_id = r.json()["id"]
    assert client.post(f"{base}/report/export", headers=h, json={}).status_code == 409   # already in progress

    with SessionLocal() as db:                       # the worker, without Celery
        export_service.build_export(db, uuid.UUID(export_id))
    st = client.get(f"{base}/report/export/{export_id}", headers=h).json()
    assert st["status"] == "ready" and st["image_count"] == 2 and st["download_url"]

    r = client.get(f"{base}/report/export/{export_id}/download", headers=h)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    assert {"images/img1.jpg", "images/img2.jpg", "labels/img1.txt", "predictions/img1.txt",
            "metadata.csv", "classes.txt"} <= names
    assert "images/img0.jpg" not in names            # not flagged
    assert len(z.read("predictions/img1.txt").decode().split()) == 6
    assert z.read("classes.txt").decode().split() == ["apple", "pear"]

    notes = client.get("/api/notifications", headers=h).json()
    assert any(n["type"] == "export_ready" for n in notes)

    # --- nothing to export -> 409 ---
    assert client.delete(f"{base}/evaluations/{ev2['id']}", headers=h).status_code == 204
    r = client.post(f"{base}/report/export", headers=h,
                    json={"include_unevaluated": False, "trust_below": 0})
    assert r.status_code == 409

    # --- delete export ---
    assert client.delete(f"{base}/report/export/{export_id}", headers=h).status_code == 204
    assert client.get(f"{base}/report/export/{export_id}", headers=h).status_code == 404