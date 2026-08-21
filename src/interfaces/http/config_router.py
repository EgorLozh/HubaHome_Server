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
            "voiceStreamPath": "/v1/voice/stream",
            "voiceTextPath": "/v1/voice/text",
            "legacyVoiceSessionPath": "/v1/voice/session",
            "metricsEnabled": settings.metrics_enabled,
            "qdrantUrl": settings.qdrant_url,
            "llmProvider": settings.llm_provider,
            "ollamaUrl": settings.ollama_url,
            "ollamaModel": settings.ollama_model,
            "deepseekModel": settings.deepseek_model,
            "deepseekApiKeyConfigured": bool(settings.deepseek_api_key),
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
            "agentMaxStepsPerTurn": settings.agent_max_steps_per_turn,
            "agentMaxToolCallsPerTurn": settings.agent_max_tool_calls_per_turn,
            "sttTimeoutMs": settings.stt_timeout_ms,
            "agentTurnTimeoutMs": settings.agent_turn_timeout_ms,
            "ttsTimeoutMs": settings.tts_timeout_ms,
            "knowledgeCollectionName": settings.knowledge_collection_name,
            "knowledgeMetadataPath": settings.knowledge_metadata_path,
        }

    return router
