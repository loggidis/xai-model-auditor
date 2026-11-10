import os
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models.auth_tokens import RefreshToken
from app.services.cleanup_service import run_cleanup
from app.services.token_service import hash_token

client = TestClient(app)


def test_cleanup_orphans_and_tokens(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)        # never touch the real storage
    old = time.time() - 3 * 86400

    def make(rel: str, mtime: float):
        d = tmp_path / rel
        d.mkdir(parents=True)
        f = d / "x.bin"
        f.write_bytes(b"data")
        for p in (f, d):
            os.utime(p, (mtime, mtime))
        return d

    user, a, b, c, d = (str(uuid.uuid4()) for _ in range(5))
    old_model = make(f"models/{user}/{a}", old)
    recent_dataset = make(f"datasets/{user}/{b}", time.time())
    old_audit = make(f"audits/{c}", old)
    old_export = make(f"exports/{d}", old)
    junk = tmp_path / "models" / "not-a-uuid"
    junk.mkdir(parents=True)

    with SessionLocal() as db:
        dry = run_cleanup(db, dry_run=True)
    assert old_model.exists() and old_audit.exists()               # nothing deleted in a dry run
    assert f"models/{user}/{a}" in dry.orphan_paths and f"audits/{c}" in dry.orphan_paths
    assert f"datasets/{user}/{b}" not in dry.orphan_paths and dry.skipped_recent >= 1

    # tokens: one expired, one still valid
    email = f"cleanup_{uuid.uuid4().hex[:8]}@example.com"
    client.post("/api/auth/register", json={"email": email, "password": "password123"})
    login = client.post("/api/auth/login", data={"username": email, "password": "password123"}).json()
    uid = uuid.UUID(client.get("/api/auth/me", headers={"Authorization": f"Bearer {login['access_token']}"}).json()["id"])
    expired_hash, valid_hash = hash_token("expired-" + uuid.uuid4().hex), hash_token("valid-" + uuid.uuid4().hex)
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        db.add(RefreshToken(user_id=uid, family_id=uuid.uuid4(), token_hash=expired_hash,
                            expires_at=now - timedelta(days=1)))
        db.add(RefreshToken(user_id=uid, family_id=uuid.uuid4(), token_hash=valid_hash,
                            expires_at=now + timedelta(days=1)))
        db.commit()

    with SessionLocal() as db:
        real = run_cleanup(db)
        assert real.refresh_tokens_deleted >= 1
        left = set(db.scalars(select(RefreshToken.token_hash).where(
            RefreshToken.token_hash.in_([expired_hash, valid_hash]))))
    assert left == {valid_hash}
    assert not old_model.exists() and not old_audit.exists() and not old_export.exists()
    assert recent_dataset.exists() and junk.exists()               # recent and non-UUID entries are left alone