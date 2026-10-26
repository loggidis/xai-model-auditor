import uuid
from typing import Literal
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.ml_model import ModelRead
from app.services import model_service

router = APIRouter(prefix="/models", tags=["models"])

@router.post("", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
def upload_model(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=255),
    description: str | None = Form(None),
    task: Literal["detection", "classification"] = Form(...),
    num_classes: int | None = Form(None, ge=1),
    input_size: int | None = Form(None, ge=1),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return model_service.create_model(db, user, file, name, description, task, num_classes, input_size)

@router.get("", response_model=list[ModelRead])
def list_models(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return model_service.list_models(db, user)

@router.get("/{model_id}", response_model=ModelRead)
def get_model(model_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return model_service.get_owned_model(db, user, model_id)

@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    model_service.delete_model(db, user, model_id)