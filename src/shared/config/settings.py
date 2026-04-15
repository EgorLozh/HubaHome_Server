from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "HubaHome Server"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    api_key: str = "dev-api-key"
    qdrant_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    whisper_model_name: str = "small"
    whisper_models_dir: str = ".models/whisper"
    whisper_auto_download: bool = True
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    piper_model_name: str = "ru_RU-irina-medium"
    piper_models_dir: str = ".models/piper"
    piper_auto_download: bool = True
    piper_use_cuda: bool = False
    piper_speaker_id: int = 0

    enable_real_stt: bool = True
    enable_real_tts: bool = True
    enable_real_web_search: bool = True

    metrics_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
