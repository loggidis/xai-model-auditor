import uuid

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.evaluation import (EvaluationCategory, EvaluationCreate, EvaluationRead,
                                    EvaluationUpdate)
from app.services import audit_service, evaluation_service

router = APIRouter(prefix="/audits", tags=["evaluations"])


@router.post("/{audit_id}/evaluations", response_model=EvaluationRead, status_code=status.HTTP_201_CREATED)
def upsert_evaluation(audit_id: uuid.UUID, data: EvaluationCreate, response: Response,
                      db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Creates the evaluation (201) or updates the existing one for the same image (200)."""
    audit = audit_service.get_owned_audit(db, user, audit_id)
    ev, created = evaluation_service.upsert_evaluation(db, user, audit, data)
    if not created:
        response.status_code = status.HTTP_200_OK
    return evaluation_service.to_read(db, ev)


@router.get("/{audit_id}/evaluations", response_model=list[EvaluationRead])
def list_evaluations(audit_id: uuid.UUID, category: EvaluationCategory | None = None,
                     limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                     db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    return evaluation_service.list_evaluations(db, user, audit, category, limit, offset)


@router.patch("/{audit_id}/evaluations/{eval_id}", response_model=EvaluationRead)
def update_evaluation(audit_id: uuid.UUID, eval_id: uuid.UUID, data: EvaluationUpdate,
                      db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    ev = evaluation_service.update_evaluation(db, user, audit, eval_id, data)
    return evaluation_service.to_read(db, ev)


@router.delete("/{audit_id}/evaluations/{eval_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_evaluation(audit_id: uuid.UUID, eval_id: uuid.UUID,
                      db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    evaluation_service.delete_evaluation(db, user, audit, eval_id)