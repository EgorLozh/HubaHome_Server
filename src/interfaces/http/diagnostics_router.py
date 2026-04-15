from fastapi import APIRouter

from src.bootstrap.container import AppContainer


def build_diagnostics_router(container: AppContainer) -> APIRouter:
    router = APIRouter()

    @router.get("/diagnostics")
    async def diagnostics() -> dict[str, object]:
        qdrant_ok = await container.vector_store_adapter.ping()
        ollama_ok = await container.llm_adapter.ping()
        return {
            "status": "ok" if qdrant_ok and ollama_ok else "degraded",
            "checks": {
                "qdrant": "up" if qdrant_ok else "down",
                "ollama": "up" if ollama_ok else "down",
            },
            "adapters": {
                "notification": container.notification_adapter.name,
                "stt": container.speech_to_text_adapter.name,
                "tts": container.text_to_speech_adapter.name,
                "webSearch": container.web_search_adapter.name,
            },
        }

    return router
