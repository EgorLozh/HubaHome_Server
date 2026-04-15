from fastapi import APIRouter

from src.bootstrap.container import AppContainer


def build_config_router(container: AppContainer) -> APIRouter:
    router = APIRouter()

    @router.get("/config")
    async def config() -> dict[str, str | bool | int]:
        settings = container.settings
        return {
            "appName": settings.app_name,
            "appEnv": settings.app_env,
            "host": settings.app_host,
            "port": settings.app_port,
            "metricsEnabled": settings.metrics_enabled,
            "qdrantUrl": settings.qdrant_url,
            "ollamaUrl": settings.ollama_url,
            "apiKeyConfigured": bool(settings.api_key),
        }

    return router
