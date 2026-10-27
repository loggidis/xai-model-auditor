import uuid
from pathlib import Path
from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.ml_model import MLModel
from app.models.user import User
from app.services import storage

ALLOWED_EXT = {".pt", ".pth"}

def create_model(db: Session, user: User, file: UploadFile, name: str,
                 description: str | None, task: str,
                 num_classes: int | None, input_size: int | None) -> MLModel:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "Only .pt / .pth files are allowed")

    model_id = uuid.uuid4()
    rel = f"models/{user.id}/{model_id}/model{ext}"
    size = storage.save_upload(file, storage.abs_path(rel),
                               settings.MAX_MODEL_SIZE_MB * 1024 * 1024)
    obj = MLModel(
        id=model_id, user_id=user.id, name=name, description=description,
        task=task, num_classes=num_classes, input_size=input_size,
        original_filename=Path(file.filename).name, file_path=rel, file_size=size,
    )
    try:
        db.add(obj)
        db.commit()
    except Exception:
        db.rollback()
        storage.remove_dir(f"models/{user.id}/{model_id}")
        raise
    db.refresh(obj)
    return obj

def list_models(db: Session, user: User) -> list[MLModel]:
    return list(db.scalars(
        select(MLModel).where(MLModel.user_id == user.id).order_by(MLModel.created_at.desc())
    ))

def get_owned_model(db: Session, user: User, model_id: uuid.UUID) -> MLModel:
    obj = db.scalar(select(MLModel).where(MLModel.id == model_id, MLModel.user_id == user.id))
    if not obj:
        raise HTTPException(404, "Model not found")
    return obj

def delete_model(db: Session, user: User, model_id: uuid.UUID) -> None:
    obj = get_owned_model(db, user, model_id)
    db.delete(obj)
    db.commit()
    storage.remove_dir(f"models/{user.id}/{model_id}")