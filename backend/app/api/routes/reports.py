import uuid

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.report import (AuditReport, ExportRead, ExportRequest, FlaggedPage, GlobalReport)
from app.services import audit_service, export_service, report_service, summary_service, pdf_service
from app.api.rate_limits import limit_by_user
from app.core.config import settings
from app.schemas.summary import SummaryRead, SummaryRequest
from typing import Literal
router = APIRouter(tags=["reports"])


@router.get("/reports", response_model=GlobalReport)
def global_report(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                  db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Aggregates over all audits of the user (Reports page of the sidebar)."""
    return report_service.build_global_report(db, user, limit, offset)


@router.get("/audits/{audit_id}/report", response_model=AuditReport)
def audit_report(audit_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    return report_service.build_audit_report(db, audit)


@router.get("/audits/{audit_id}/report/flagged", response_model=FlaggedPage)
def flagged(audit_id: uuid.UUID,
            trust_below: int = Query(50, ge=0, le=100, description="evaluated 'correct' images below this trust are flagged; 0 disables"),
            include_unevaluated: bool = Query(True, description="also flag unevaluated images tagged needs_review"),
            limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
            db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    items = report_service.flagged_items(db, audit, trust_below, include_unevaluated)
    return FlaggedPage(items=[report_service.to_flagged_item(r) for r in items[offset:offset + limit]],
                       total=len(items), limit=limit, offset=offset)


# ---------- export (background job) ----------

@router.post("/audits/{audit_id}/report/export", response_model=ExportRead,
             status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(limit_by_user("jobs", settings.RATE_LIMIT_JOBS_PER_HOUR, 3600))])
def create_export(audit_id: uuid.UUID, data: ExportRequest | None = None,
                  db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    return export_service.to_read(export_service.create_export(db, user, audit, data or ExportRequest()))


@router.get("/audits/{audit_id}/report/export", response_model=list[ExportRead])
def list_exports(audit_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    return [export_service.to_read(e) for e in export_service.list_exports(db, audit)]


@router.get("/audits/{audit_id}/report/export/{export_id}", response_model=ExportRead)
def get_export(audit_id: uuid.UUID, export_id: uuid.UUID,
               db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    return export_service.to_read(export_service.get_export(db, audit, export_id))


@router.get("/audits/{audit_id}/report/export/{export_id}/download")
def download_export(audit_id: uuid.UUID, export_id: uuid.UUID,
                    db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    path = export_service.export_file(export_service.get_export(db, audit, export_id))
    return FileResponse(path, media_type="application/zip", filename=f"flagged_{str(audit_id)[:8]}.zip")


@router.delete("/audits/{audit_id}/report/export/{export_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_export(audit_id: uuid.UUID, export_id: uuid.UUID,
                  db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    export_service.delete_export(db, audit, export_id)

# ---------- LLM summary (background job) ----------

@router.post("/audits/{audit_id}/report/summary-llm", response_model=SummaryRead,
             status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(limit_by_user("jobs", settings.RATE_LIMIT_JOBS_PER_HOUR, 3600))])
def request_summary(audit_id: uuid.UUID, data: SummaryRequest | None = None,
                    db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Starts the generation of an LLM summary of the report. Poll the GET endpoint for the result."""
    audit = audit_service.get_owned_audit(db, user, audit_id)
    audit = summary_service.request_summary(db, user, audit, data or SummaryRequest())
    return summary_service.to_read(audit)


@router.get(
    "/audits/{audit_id}/report/summary-llm",
    response_model=SummaryRead,
    responses={200: {
        "description": "JSON status and summary (default) or, with `format=pdf`, the summary as a PDF file",
        "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
    }},
)
def get_summary(audit_id: uuid.UUID,
                fmt: Literal["json", "pdf"] = Query("json", alias="format",
                                                    description="`pdf` downloads the finished summary as a PDF"),
                db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    audit = audit_service.get_owned_audit(db, user, audit_id)
    if fmt == "pdf":
        data, filename = pdf_service.summary_pdf(db, audit)
        return Response(content=data, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    return summary_service.to_read(audit)