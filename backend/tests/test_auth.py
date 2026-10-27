from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_register_login_me():
    email = "test_user@example.com"
    client.post("/api/auth/register", json={"email": email, "password": "password123"})
    r = client.post("/api/auth/login", data={"username": email, "password": "password123"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json()["email"] == email