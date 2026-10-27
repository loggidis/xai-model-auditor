from fastapi import APIRouter, Depends, HTTPException, Query
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.xai import MaskRead, XAIMethodRead
from app.xai import registry
from app.xai.masks import MASKS

router = APIRouter(prefix="/xai", tags=["xai"])


@router.get("/methods", response_model=list[XAIMethodRead])
def list_methods(architecture: str | None = Query(None, description="π.χ. yolo_detect"),
                 user: User = Depends(get_current_user)):
    return [m.metadata() for m in registry.list_methods(architecture)]


@router.get("/methods/{method}/masks", response_model=list[MaskRead])
def list_masks(method: str, user: User = Depends(get_current_user)):
    try:
        m = registry.get_method(method)
    except KeyError:
        raise HTTPException(404, "Method not found")
    return [MASKS[n].metadata() for n in m.supported_masks]