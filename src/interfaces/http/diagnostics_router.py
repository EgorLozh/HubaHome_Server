from fastapi import APIRouter

from src.bootstrap.container import AppContainer


def build_diagnostics_router(container: AppContainer) -> APIRouter:
    router = APIRouter()

    @router.get("/diagnostics")
    async def diagnostics() -> dict[str, object]:
        qdrant_ok = await container.vector_store_adapter.ping()
        ollama_ok = await container.llm_adapter.ping()
        whisper_model_status = container.whisper_model_status
        piper_model_status = container.piper_model_status
        return {
            "status": "ok" if qdrant_ok and ollama_ok else "degraded",
            "checks": {
                "qdrant": "up" if qdrant_ok else "down",
                "ollama": "up" if ollama_ok else "down",
                "whisperModel": whisper_model_status,
                "piperModel": piper_model_status,
            },
            "adapters": {
                "notification": container.notification_adapter.name,
                "stt": container.speech_to_text_adapter.name,
                "tts": container.text_to_speech_adapter.name,
                "webSearch": container.web_search_adapter.name,
            },
            "piper": {
                "modelName": container.settings.piper_model_name,
                "modelsDir": container.settings.piper_models_dir,
                "autoDownload": container.settings.piper_auto_download,
                "status": piper_model_status,
            },
            "whisper": {
                "modelName": container.settings.whisper_model_name,
                "modelsDir": container.settings.whisper_models_dir,
                "autoDownload": container.settings.whisper_auto_download,
                "device": container.settings.whisper_device,
                "computeType": container.settings.whisper_compute_type,
                "status": whisper_model_status,
            },
        }

    return router
