import uuid
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.dataset import DatasetRead, ImagePage
from app.services import dataset_service
from fastapi.responses import FileResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=255),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return dataset_service.create_dataset(db, user, file, name)

@router.get("", response_model=list[DatasetRead])
def list_datasets(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return dataset_service.list_datasets(db, user)

@router.get("/{dataset_id}", response_model=DatasetRead)
def get_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return dataset_service.get_owned_dataset(db, user, dataset_id)

@router.get("/{dataset_id}/images", response_model=ImagePage)
def list_images(
    dataset_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ds = dataset_service.get_owned_dataset(db, user, dataset_id)
    items, total = dataset_service.list_images(db, ds, limit, offset)
    return ImagePage(items=items, total=total, limit=limit, offset=offset)

@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    dataset_service.delete_dataset(db, user, dataset_id)

@router.get("/{dataset_id}/images/{image_id}")
def get_image(dataset_id: uuid.UUID, image_id: uuid.UUID,
              db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    ds = dataset_service.get_owned_dataset(db, user, dataset_id)
    return FileResponse(dataset_service.get_image_file(db, ds, image_id))