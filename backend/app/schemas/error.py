from typing import Any

from pydantic import BaseModel


class ErrorBody(BaseModel):
    code: str                       # machine readable, e.g. "not_found", "validation_error", "rate_limited"
    message: str                    # human readable
    details: Any | None = None      # validation errors: list of {field, message}


class ErrorResponse(BaseModel):
    error: ErrorBody