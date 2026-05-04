import json
import os
from pathlib import Path

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
        metadata_file = Path(container.settings.knowledge_metadata_path)
        metadata_json_valid = _is_metadata_json_valid(metadata_file)
        metadata_path_writable = _is_metadata_path_writable(metadata_file)
        runtime_budget_ok = (
            container.settings.agent_max_steps_per_turn > 0
            and container.settings.agent_max_tool_calls_per_turn > 0
            and container.settings.stt_timeout_ms >= 1000
            and container.settings.agent_turn_timeout_ms >= 1000
            and container.settings.tts_timeout_ms >= 1000
        )
        preflight_ok = metadata_json_valid and metadata_path_writable and runtime_budget_ok
        overall_ok = qdrant_ok and ollama_ok and preflight_ok
        return {
            "status": "ok" if overall_ok else "degraded",
            "checks": {
                "qdrant": "up" if qdrant_ok else "down",
                "ollama": "up" if ollama_ok else "down",
                "whisperModel": whisper_model_status,
                "piperModel": piper_model_status,
            },
            "preflight": {
                "metadataPathWritable": metadata_path_writable,
                "metadataJsonValid": metadata_json_valid,
                "runtimeBudgetValid": runtime_budget_ok,
                "agentMaxStepsPerTurn": container.settings.agent_max_steps_per_turn,
                "agentMaxToolCallsPerTurn": container.settings.agent_max_tool_calls_per_turn,
                "sttTimeoutMs": container.settings.stt_timeout_ms,
                "agentTurnTimeoutMs": container.settings.agent_turn_timeout_ms,
                "ttsTimeoutMs": container.settings.tts_timeout_ms,
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


def _is_metadata_json_valid(metadata_file: Path) -> bool:
    if not metadata_file.exists():
        return True
    try:
        content = metadata_file.read_text(encoding="utf-8").strip()
        if not content:
            return True
        payload = json.loads(content)
        return isinstance(payload, dict)
    except Exception:
        return False


def _is_metadata_path_writable(metadata_file: Path) -> bool:
    try:
        parent = metadata_file.parent
        parent.mkdir(parents=True, exist_ok=True)
        if metadata_file.exists():
            return os.access(metadata_file, os.W_OK)
        return os.access(parent, os.W_OK)
    except Exception:
        return False
