from fastapi import HTTPException

from src.shared.errors.base import AppError


def to_http_exception(error: AppError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.message)
