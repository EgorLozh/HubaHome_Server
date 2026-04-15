from fastapi import APIRouter

from src.bootstrap.container import AppContainer
from src.interfaces.http.config_router import build_config_router
from src.interfaces.http.diagnostics_router import build_diagnostics_router
from src.interfaces.http.health_router import router as health_router
from src.shared.observability.metrics import metrics_response


def build_api_router(container: AppContainer) -> APIRouter:
    router = APIRouter()
    router.include_router(health_router)
    router.include_router(build_config_router(container))
    router.include_router(build_diagnostics_router(container))

    @router.get("/metrics")
    async def metrics():
        return metrics_response()

    return router
