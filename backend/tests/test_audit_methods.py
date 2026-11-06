import io
import uuid
import zipfile
from dataclasses import asdict

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("ultralytics")
from ultralytics import YOLO

from app.db.session import SessionLocal
from app.main import app
from app.models.ml_model import MLModel
from app.worker import tasks
from app.xai.loader import from_yolo

client = TestClient(app)


def _auth():
    email = "methods_user@example.com"
    client.post("/api/auth/register", json={"email": email, "password": "password123"})
    r = client.post("/api/auth/login", data={"username": email, "password": "password123"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _jpg(seed: int) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_two_methods_agreement_and_tag(monkeypatch):
    h = _auth()
    me = client.get("/api/auth/me", headers=h).json()
    tiny = from_yolo(YOLO("yolov8n.yaml"))
    monkeypatch.setattr("app.xai.loader.load_model", lambda path: tiny)
    monkeypatch.setattr(tasks.run_audit, "delay", lambda *a, **k: type("R", (), {"id": "test"})())

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
        for i in range(2):
            z.writestr(f"img{i}.jpg", _jpg(i))
    buf.seek(0)
    ds = client.post("/api/datasets", headers=h, data={"name": "t"},
                     files={"file": ("d.zip", buf, "application/zip")}).json()

    base = {"model_id": model_id, "dataset_id": ds["id"], "conf_threshold": 0.01}
    # negative validation
    bad = [
        {"xai_methods": ["gradcam_pp"], "method_params": {"lime_image": {"num_samples": 20}}},
        {"xai_methods": ["lime_image"], "method_params": {"lime_image": {"nope": 1}}},
        {"xai_methods": ["lime_image"], "method_params": {"lime_image": {"num_samples": 1}}},
        {"xai_methods": ["gradcam_pp"], "tagging": {"agreement_low": 0.9, "agreement_high": 0.2}},
    ]
    for extra in bad:
        assert client.post("/api/audits", headers=h, json={**base, **extra}).status_code == 422

    r = client.post("/api/audits", headers=h, json={
        **base, "xai_methods": ["gradcam_pp", "lime_image"],
        "method_params": {"lime_image": {"num_samples": 20, "n_segments": 10}},
        "tagging": {"agreement_high": 0.6},
    })
    assert r.status_code == 201
    audit_id = r.json()["id"]
    assert r.json()["config"]["tagging"]["agreement_high"] == 0.6

    with SessionLocal() as db:
        tasks._run(db, uuid.UUID(audit_id))

    res = client.get(f"/api/audits/{audit_id}/results?status=done", headers=h).json()
    assert res["total"] == 2 and all(i["auto_tag"] for i in res["items"])
    detail = client.get(f"/api/audits/{audit_id}/results/{res['items'][0]['image_id']}", headers=h).json()
    assert detail["auto_tag"] and detail["auto_tag_reason"]
    assert {h_["method"] for h_ in detail["heatmaps"]} == {"gradcam_pp", "lime_image"}
    if detail["auto_tag"] != "no_detection":            # with random weights there may be no targets
        assert detail["agreement_details"]["pairs"][0]["methods"] == ["gradcam_pp", "lime_image"]

    by_tag = client.get(f"/api/audits/{audit_id}/results?auto_tag={detail['auto_tag']}", headers=h).json()
    assert by_tag["total"] >= 1

    # cleanup
    assert client.delete(f"/api/audits/{audit_id}", headers=h).status_code == 204
    assert client.delete(f"/api/datasets/{ds['id']}", headers=h).status_code == 204
    assert client.delete(f"/api/models/{model_id}", headers=h).status_code == 204