import re
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from fastapi import HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.dataset import Dataset, DatasetImage
from app.models.user import User
from app.services import storage

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAFE_NAME = re.compile(r"[\w.\- ()]+")  # όχι ":", "/", "\" κλπ.

def _extract_zip(zip_path: Path, dest: Path) -> tuple[list[str], set[str]]:
    images_dir, labels_dir = dest / "images", dest / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    max_total = settings.MAX_DATASET_UNCOMPRESSED_MB * 1024 * 1024
    images: list[str] = []
    label_stems: set[str] = set()
    seen: set[tuple[str, str]] = set()
    total = 0

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid zip file")

    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > settings.MAX_DATASET_FILES:
            raise HTTPException(413, "Too many files in zip")
        if sum(i.file_size for i in infos) > max_total:
            raise HTTPException(413, "Zip content too large")

        for info in infos:
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            if not parts or "__MACOSX" in parts:
                continue
            name = parts[-1]
            if name.startswith(".") or not SAFE_NAME.fullmatch(name):
                continue
            ext = Path(name).suffix.lower()

            if ext in IMAGE_EXT:
                kind, target = "img", images_dir / name
            elif ext == ".txt" and name.lower() != "classes.txt":
                kind, target = "lbl", labels_dir / name
            else:
                continue

            if (kind, name) in seen:
                raise HTTPException(400, f"Duplicate filename in zip: {name}")
            seen.add((kind, name))

            with zf.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_total:
                        raise HTTPException(413, "Zip content too large")
                    dst.write(chunk)

            if kind == "img":
                images.append(name)
            else:
                label_stems.add(Path(name).stem)

    if not images:
        raise HTTPException(400, "No images found in zip")
    return images, label_stems

def create_dataset(db: Session, user: User, file: UploadFile, name: str) -> Dataset:
    if Path(file.filename or "").suffix.lower() != ".zip":
        raise HTTPException(400, "Only .zip files are allowed")

    dataset_id = uuid.uuid4()
    rel_dir = f"datasets/{user.id}/{dataset_id}"
    root = storage.abs_path(rel_dir)
    zip_path = root / "upload.zip"

    try:
        storage.save_upload(file, zip_path, settings.MAX_DATASET_ZIP_MB * 1024 * 1024)
        images, label_stems = _extract_zip(zip_path, root)
        zip_path.unlink(missing_ok=True)

        rows = [
            DatasetImage(dataset_id=dataset_id, filename=n, has_label=Path(n).stem in label_stems)
            for n in sorted(images)
        ]
        dataset = Dataset(
            id=dataset_id, user_id=user.id, name=name, path=rel_dir,
            image_count=len(rows), label_count=sum(r.has_label for r in rows),
        )
        db.add(dataset)
        db.flush()            # δημιουργεί πρώτα το dataset (FK)
        db.add_all(rows)
        db.commit()
    except Exception:
        db.rollback()
        storage.remove_dir(rel_dir)   # καθαρισμός μισοτελειωμένου upload
        raise

    db.refresh(dataset)
    return dataset

def list_datasets(db: Session, user: User) -> list[Dataset]:
    return list(db.scalars(
        select(Dataset).where(Dataset.user_id == user.id).order_by(Dataset.created_at.desc())
    ))

def get_owned_dataset(db: Session, user: User, dataset_id: uuid.UUID) -> Dataset:
    obj = db.scalar(select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user.id))
    if not obj:
        raise HTTPException(404, "Dataset not found")
    return obj

def list_images(db: Session, dataset: Dataset, limit: int, offset: int):
    total = db.scalar(select(func.count()).select_from(DatasetImage)
                      .where(DatasetImage.dataset_id == dataset.id))
    items = list(db.scalars(
        select(DatasetImage).where(DatasetImage.dataset_id == dataset.id)
        .order_by(DatasetImage.filename).limit(limit).offset(offset)
    ))
    return items, total

def delete_dataset(db: Session, user: User, dataset_id: uuid.UUID) -> None:
    obj = get_owned_dataset(db, user, dataset_id)
    rel = obj.path
    db.delete(obj)
    db.commit()
    storage.remove_dir(rel)