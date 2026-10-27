import shutil
from pathlib import Path
from fastapi import HTTPException, UploadFile
from app.core.config import settings

def abs_path(rel: str) -> Path:
    root = settings.STORAGE_DIR.resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Invalid storage path")
    return p

def save_upload(file: UploadFile, dest: Path, max_bytes: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with open(dest, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, "File too large")
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return written

def remove_dir(rel: str) -> None:
    shutil.rmtree(abs_path(rel), ignore_errors=True)