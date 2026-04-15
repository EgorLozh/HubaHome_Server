from contextvars import ContextVar
from uuid import uuid4


_correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")


def set_correlation_id(value: str | None = None) -> str:
    correlation_id = value or str(uuid4())
    _correlation_id_ctx.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    return _correlation_id_ctx.get() or set_correlation_id()
