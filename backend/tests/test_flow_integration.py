import io
import uuid
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("ultralytics")
from ultralytics import YOLO

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.services import export_service, llm_service, summary_service
from app.worker import export_tasks, summary_tasks, tasks
from app.xai.loader import from_yolo

client = TestClient(app)


def _jpg(seed: int) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_full_flow(monkeypatch):
    # ---- patches: tiny random-weight model, no Celery, fake LLM (no tokens are spent) ----
    tiny = from_yolo(YOLO("yolov8n.yaml"))
    monkeypatch.setattr("app.xai.loader.load_model", lambda path: tiny)
    fake_delay = lambda *a, **k: type("R", (), {"id": "test"})()
    monkeypatch.setattr(tasks.run_audit, "delay", fake_delay)
    monkeypatch.setattr(export_tasks.run_export, "delay", fake_delay)
    monkeypatch.setattr(summary_tasks.run_summary, "delay", fake_delay)
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    calls: list[list[dict]] = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        return "## Overview\nEverything looks fine.", "test-model"

    monkeypatch.setattr(llm_service, "chat_completion", fake_chat)

    # ---- register, login, settings ----
    email = f"flow_{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/register", json={"email": email, "password": "password123"})
    assert r.status_code == 201 and r.json()["email_verified"] is False
    login = client.post("/api/auth/login", data={"username": email, "password": "password123"})
    assert login.status_code == 200 and login.json()["refresh_token"]
    h = {"Authorization": f"Bearer {login.json()['access_token']}"}
    assert client.patch("/api/settings", headers=h, json={"batch_size": 2}).json()["batch_size"] == 2

    # ---- upload model (the metadata comes from the patched loader) ----
    r = client.post("/api/models", headers=h, data={"name": "m", "task": "detection"},
                    files={"file": ("model.pt", b"not-a-real-model", "application/octet-stream")})
    assert r.status_code == 201
    model = r.json()
    assert model["architecture_type"] == "yolo_detect" and model["class_names"]
    layers = client.get(f"/api/models/{model['id']}/recommended-layers", headers=h).json()
    assert sum(l["default"] for l in layers) == 1
    methods = {m["name"] for m in client.get("/api/xai/methods", headers=h).json()}
    assert {"gradcam_pp", "integrated_gradients"} <= methods

    # ---- upload dataset ----
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(4):
            z.writestr(f"img{i}.jpg", _jpg(i))
        z.writestr("labels/img0.txt", "0 0.5 0.5 0.2 0.2")
        z.writestr("labels/img1.txt", "1 0.5 0.5 0.2 0.2")
    buf.seek(0)
    ds = client.post("/api/datasets", headers=h, data={"name": "d"},
                     files={"file": ("d.zip", buf, "application/zip")}).json()
    assert ds["image_count"] == 4 and ds["label_count"] == 2

    # ---- audit ----
    r = client.post("/api/audits", headers=h, json={
        "model_id": model["id"], "dataset_id": ds["id"],
        "xai_methods": ["gradcam_pp", "integrated_gradients"], "conf_threshold": 0.01,
        "method_params": {"integrated_gradients": {"n_steps": 4, "batch_size": 2, "smooth_sigma": 2.0}},
    })
    assert r.status_code == 201 and r.json()["progress"]["pending"] == 4
    audit_id = r.json()["id"]
    base = f"/api/audits/{audit_id}"
    with SessionLocal() as db:                                   # the audit worker, without Celery
        tasks._run(db, uuid.UUID(audit_id))
    audit = client.get(base, headers=h).json()
    assert audit["status"] == "completed" and audit["progress"]["done"] == 4

    done = client.get(f"{base}/results?status=done&limit=50", headers=h).json()["items"]
    assert len(done) == 4 and all(i["auto_tag"] for i in done)
    detail = client.get(f"{base}/results/{done[0]['image_id']}", headers=h).json()
    assert {x["method"] for x in detail["heatmaps"]} == {"gradcam_pp", "integrated_gradients"}
    png = client.get(f"{base}/results/{done[0]['image_id']}/heatmap/gradcam_pp", headers=h)
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"

    # ---- evaluations ----
    first, second = done[0], done[1]
    assert client.post(f"{base}/evaluations", headers=h, json={
        "image_id": first["image_id"], "trust_score": 90, "category": "correct_right_reason"}).status_code == 201
    r = client.post(f"{base}/evaluations", headers=h, json={
        "image_id": second["image_id"], "trust_score": 10, "category": "wrong_prediction",
        "notes": "secret-note-xyz"})
    assert r.status_code == 201

    # ---- report and flagged ----
    rep = client.get(f"{base}/report", headers=h).json()
    assert rep["processed_images"] == 4 and rep["evaluated_images"] == 2
    cats = {s["key"]: s["count"] for s in rep["by_category"]}
    assert cats["correct_right_reason"] == 1 and cats["wrong_prediction"] == 1 and cats["unevaluated"] == 2
    flagged = client.get(f"{base}/report/flagged?limit=200", headers=h).json()
    assert second["image_id"] in {i["image_id"] for i in flagged["items"]}
    scores = [i["sort_score"] for i in flagged["items"] if i["sort_score"] is not None]
    assert scores == sorted(scores)

    # ---- LLM summary (async job, fake provider) ----
    r = client.post(f"{base}/report/summary-llm", headers=h, json={"language": "en", "top_flagged": 5})
    assert r.status_code == 202 and r.json()["status"] == "pending"
    assert client.post(f"{base}/report/summary-llm", headers=h, json={}).status_code == 409   # already in progress
    # the PDF is not available until the summary is done
    assert client.get(f"{base}/report/summary-llm?format=pdf", headers=h).status_code == 409
    with SessionLocal() as db:
        summary_service.generate_summary(db, uuid.UUID(audit_id))
    s = client.get(f"{base}/report/summary-llm", headers=h).json()
    assert s["status"] == "done" and "Overview" in s["summary"] and s["model"] == "test-model"
    assert len(calls) == 1
    pdf = client.get(f"{base}/report/summary-llm?format=pdf", headers=h)
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF") and "attachment" in pdf.headers["content-disposition"]
    prompt = calls[0][-1]["content"]
    assert "secret-note-xyz" not in prompt and "img0.jpg" not in prompt     # notes and file names are not sent

    # ---- dry-run mode: the whole flow runs, but the LLM is never called ----
    monkeypatch.setattr(settings, "LLM_DRY_RUN", True)
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)                     # no key needed in dry-run
    r = client.post(f"{base}/report/summary-llm", headers=h, json={"top_flagged": 3})
    assert r.status_code == 202
    with SessionLocal() as db:
        summary_service.generate_summary(db, uuid.UUID(audit_id))
    s = client.get(f"{base}/report/summary-llm", headers=h).json()
    assert s["status"] == "done" and s["model"] == "dry-run" and s["summary"].startswith("[DRY RUN]")
    assert "secret-note-xyz" not in s["summary"] and "img0.jpg" not in s["summary"]
    assert len(calls) == 1                                                  # still only the first (fake) call
    monkeypatch.setattr(settings, "LLM_DRY_RUN", False)

    # ---- export (async job) ----
    r = client.post(f"{base}/report/export", headers=h, json={})
    assert r.status_code == 202
    export_id = r.json()["id"]
    with SessionLocal() as db:
        export_service.build_export(db, uuid.UUID(export_id))
    assert client.get(f"{base}/report/export/{export_id}", headers=h).json()["status"] == "ready"
    dl = client.get(f"{base}/report/export/{export_id}/download", headers=h)
    assert dl.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(dl.content)).namelist())
    assert {"metadata.csv", "classes.txt"} <= names

    # ---- notifications and global report ----
    types = {n["type"] for n in client.get("/api/notifications?limit=100", headers=h).json()}
    assert {"audit_completed", "summary_ready", "export_ready"} <= types
    g = client.get("/api/reports", headers=h).json()
    assert any(a["audit_id"] == audit_id for a in g["audits"])

    # ---- cleanup of everything created ----
    assert client.delete(f"{base}/report/export/{export_id}", headers=h).status_code == 204
    assert client.delete(base, headers=h).status_code == 204
    gone = client.get(base, headers=h)
    assert gone.status_code == 404 and gone.json()["error"]["code"] == "not_found"
    assert client.delete(f"/api/datasets/{ds['id']}", headers=h).status_code == 204
    assert client.delete(f"/api/models/{model['id']}", headers=h).status_code == 204