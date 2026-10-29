import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.audit import AuditCreate, AuditRead, ResultDetail, ResultPage
from app.services import audit_service

router = APIRouter(prefix="/audits", tags=["audits"])


@router.post("", response_model=AuditRead, status_code=status.HTTP_201_CREATED)
def create_audit(data: AuditCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.create_audit(db, user, data)
    return audit_service.to_read(db, [audit])[0]


@router.get("", response_model=list[AuditRead])
def list_audits(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return audit_service.list_audits(db, user, limit, offset)


@router.get("/{audit_id}", response_model=AuditRead)
def get_audit(audit_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    return audit_service.to_read(db, [audit])[0]


@router.delete("/{audit_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_audit(audit_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit_service.delete_audit(db, user, audit_id)


@router.get("/{audit_id}/results", response_model=ResultPage)
def list_results(
    audit_id: uuid.UUID,
    status: Literal["pending", "processing", "done", "failed"] | None = None,
    batch_index: int | None = Query(None, ge=0),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db), user: User = Depends(get_current_user),
):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    items, total = audit_service.list_results(db, audit, status, batch_index, limit, offset)
    return ResultPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/{audit_id}/results/{image_id}", response_model=ResultDetail)
def get_result(audit_id: uuid.UUID, image_id: uuid.UUID,
               db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    return audit_service.get_result_detail(db, audit, image_id)


@router.get("/{audit_id}/results/{image_id}/heatmap/{method}")
def get_heatmap(audit_id: uuid.UUID, image_id: uuid.UUID, method: str,
                raw: bool = Query(False, description="true = grayscale heatmap χωρίς overlay"),
                db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    path = audit_service.get_heatmap_file(db, audit, image_id, method, raw)
    return FileResponse(path, media_type="image/png")