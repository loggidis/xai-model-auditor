import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import engine
from app.models.audit import Audit
from app.models.audit_export import AuditExport
from app.models.auth_tokens import EmailVerificationToken, RefreshToken
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.services import storage

log = logging.getLogger(__name__)
LOCK_ID = 8_000_001            # Postgres advisory lock: only one cleanup at a time
MAX_LISTED = 100


@dataclass
class CleanupReport:
    dry_run: bool
    refresh_tokens_deleted: int = 0
    verification_tokens_deleted: int = 0
    orphan_paths: list[str] = field(default_factory=list)      # relative to the storage root
    orphan_bytes: int = 0
    skipped_recent: int = 0                                    # orphans younger than the grace period
    missing_files: list[str] = field(default_factory=list)     # report only, never deleted
    errors: list[str] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid_dirs(path: Path):
    """Yields the sub-directories whose name is a UUID (anything else is ignored)."""
    if not path.is_dir():
        return
    for p in sorted(path.iterdir()):
        if p.is_dir():
            try:
                uuid.UUID(p.name)
            except ValueError:
                continue
            yield p


def _newest_mtime(path: Path) -> float:
    newest = path.stat().st_mtime
    try:
        for child in path.iterdir():
            newest = max(newest, child.stat().st_mtime)
    except OSError:
        pass
    return newest


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class _Sweeper:
    def __init__(self, root: Path, cutoff: float, report: CleanupReport):
        self.root, self.cutoff, self.report = root, cutoff, report

    def remove(self, path: Path) -> None:
        if _newest_mtime(path) > self.cutoff:       # recent: it may be an upload still in progress
            self.report.skipped_recent += 1
            return
        self.report.orphan_bytes += _size(path)
        self.report.orphan_paths.append(path.relative_to(self.root).as_posix())
        if self.report.dry_run:
            return
        try:
            shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
        except OSError as e:
            self.report.errors.append(f"{path.name}: {e}")

    def remove_if_empty(self, path: Path) -> None:
        if self.report.dry_run:
            return
        try:
            if path.is_dir() and not any(path.iterdir()) and _newest_mtime(path) <= self.cutoff:
                path.rmdir()
        except OSError:
            pass


def _sweep_two_levels(sw: _Sweeper, base: Path, known: set[tuple[str, str]]) -> None:
    for user_dir in _uuid_dirs(base):
        for entity_dir in _uuid_dirs(user_dir):
            if (user_dir.name.lower(), entity_dir.name.lower()) not in known:
                sw.remove(entity_dir)
        sw.remove_if_empty(user_dir)


def _cleanup_tokens(db: Session, report: CleanupReport) -> None:
    now = _now()
    # Revoked tokens that have not expired yet are kept: they are needed for reuse detection.
    refresh_cond = RefreshToken.expires_at < now
    verify_cond = or_(
        EmailVerificationToken.expires_at < now,
        and_(EmailVerificationToken.used_at.is_not(None),
             EmailVerificationToken.used_at < now - timedelta(days=7)),
    )
    report.refresh_tokens_deleted = db.scalar(
        select(func.count()).select_from(RefreshToken).where(refresh_cond)) or 0
    report.verification_tokens_deleted = db.scalar(
        select(func.count()).select_from(EmailVerificationToken).where(verify_cond)) or 0
    if not report.dry_run:
        db.execute(delete(RefreshToken).where(refresh_cond))
        db.execute(delete(EmailVerificationToken).where(verify_cond))
        db.commit()


def _cleanup_files(db: Session, report: CleanupReport) -> None:
    root = settings.STORAGE_DIR.resolve()
    sw = _Sweeper(root, time.time() - settings.CLEANUP_ORPHAN_GRACE_HOURS * 3600, report)

    model_keys = {(str(u), str(i)) for u, i in db.execute(select(MLModel.user_id, MLModel.id))}
    dataset_keys = {(str(u), str(i)) for u, i in db.execute(select(Dataset.user_id, Dataset.id))}
    audit_ids = {str(i) for i in db.scalars(select(Audit.id))}
    export_ids = {str(i) for i in db.scalars(select(AuditExport.id))}

    _sweep_two_levels(sw, root / "models", model_keys)
    _sweep_two_levels(sw, root / "datasets", dataset_keys)

    for audit_dir in _uuid_dirs(root / "audits"):
        if audit_dir.name.lower() not in audit_ids:
            sw.remove(audit_dir)

    for audit_dir in _uuid_dirs(root / "exports"):
        if audit_dir.name.lower() not in audit_ids:
            sw.remove(audit_dir)
            continue
        for f in sorted(audit_dir.iterdir()):
            if not f.is_file():
                continue
            if f.name.endswith(".zip.part"):          # an interrupted build (old ones only, see the grace period)
                sw.remove(f)
            elif f.name.endswith(".zip") and f.name[:-4].lower() not in export_ids:
                sw.remove(f)


def _report_missing_files(db: Session, report: CleanupReport) -> None:
    for model_id, path in db.execute(select(MLModel.id, MLModel.file_path)):
        if not storage.abs_path(path).is_file():
            report.missing_files.append(f"model:{model_id}")
    for dataset_id, path in db.execute(select(Dataset.id, Dataset.path)):
        if not storage.abs_path(path).is_dir():
            report.missing_files.append(f"dataset:{dataset_id}")
    for export_id, path in db.execute(
            select(AuditExport.id, AuditExport.file_path).where(AuditExport.status == "ready")):
        if path and not storage.abs_path(path).is_file():
            report.missing_files.append(f"export:{export_id}")
    report.missing_files = report.missing_files[:MAX_LISTED]


def run_cleanup(db: Session, dry_run: bool = False) -> CleanupReport:
    report = CleanupReport(dry_run=dry_run)
    # The advisory lock is session level, so it must live on a dedicated connection for the whole run.
    with engine.connect() as lock_conn:
        if not lock_conn.execute(select(func.pg_try_advisory_lock(LOCK_ID))).scalar():
            report.errors.append("Another cleanup is already running")
            return report
        try:
            _cleanup_tokens(db, report)
            _cleanup_files(db, report)
            _report_missing_files(db, report)
        finally:
            lock_conn.execute(select(func.pg_advisory_unlock(LOCK_ID)))
            lock_conn.commit()
    report.orphan_paths = report.orphan_paths[:MAX_LISTED * 5]
    return report