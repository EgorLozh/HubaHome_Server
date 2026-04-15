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
            "ollamaModel": settings.ollama_model,
            "realSttEnabled": settings.enable_real_stt,
            "realTtsEnabled": settings.enable_real_tts,
            "realWebSearchEnabled": settings.enable_real_web_search,
            "apiKeyConfigured": bool(settings.api_key),
            "whisperModelName": settings.whisper_model_name,
            "whisperModelsDir": settings.whisper_models_dir,
            "whisperAutoDownload": settings.whisper_auto_download,
            "whisperDevice": settings.whisper_device,
            "whisperComputeType": settings.whisper_compute_type,
            "piperModelName": settings.piper_model_name,
            "piperModelsDir": settings.piper_models_dir,
            "piperAutoDownload": settings.piper_auto_download,
            "piperUseCuda": settings.piper_use_cuda,
        }

    return router
