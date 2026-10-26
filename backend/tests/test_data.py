import io
import zipfile
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def _auth():
    email = "data_user@example.com"
    client.post("/api/auth/register", json={"email": email, "password": "password123"})
    r = client.post("/api/auth/login", data={"username": email, "password": "password123"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}

def test_dataset_upload_and_delete():
    h = _auth()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("images/a.jpg", b"fake")
        z.writestr("labels/a.txt", "0 0.5 0.5 0.2 0.2")
        z.writestr("../evil.jpg", b"fake")   # το path αγνοείται, κρατιέται μόνο το όνομα
    buf.seek(0)
    r = client.post("/api/datasets", headers=h, data={"name": "t"},
                    files={"file": ("d.zip", buf, "application/zip")})
    assert r.status_code == 201
    body = r.json()
    assert body["image_count"] == 2 and body["label_count"] == 1
    assert client.delete(f"/api/datasets/{body['id']}", headers=h).status_code == 204